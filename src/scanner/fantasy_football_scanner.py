"""
Fantasy Football scanner for market data collection

This module implements a scanner for fantasy football data sources,
providing access to player stats, news, and analysis from various
fantasy football platforms and news sources.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse

from .base_scanner import BaseScanner, MarketMention


class FantasyFootballScanner(BaseScanner):
    """
    Scanner for Fantasy Football data sources
    
    This scanner fetches fantasy football content from various sources
    including RSS feeds, news APIs, and fantasy football platforms.
    """
    
    def __init__(self):
        """Initialize the Fantasy Football scanner"""
        super().__init__("Fantasy Football", rate_limit=1.5)
        
        # Fantasy football RSS feeds and sources
        self.feeds = [
            "https://feeds.feedburner.com/espn/fantasy/football",
            "https://www.fantasypros.com/feed/",
            "https://www.sleeper.app/api/v1/news",
            "https://www.fantasyalarm.com/rss/fantasy-football",
            "https://www.rotoworld.com/rss/fantasy-football",
            "https://www.numberfire.com/rss/fantasy-football",
            "https://www.fantasyfootballtoday.com/rss",
            "https://www.thefantasyfootballers.com/feed/"
        ]
        
        # Fantasy football specific keywords for relevance scoring
        self.fantasy_keywords = [
            "fantasy football", "fantasy", "draft", "waiver wire", "start/sit",
            "sleepers", "busts", "injury report", "player news", "trade value",
            "projections", "rankings", "matchup", "DFS", "daily fantasy",
            "lineup", "roster", "free agent", "trade", "pickup"
        ]
        
        logger.info("Initialized Fantasy Football scanner")
    
    async def scan(self, keywords: List[str], max_results: int = 100) -> List[MarketMention]:
        """
        Scan fantasy football sources for mentions
        
        Args:
            keywords: Keywords to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of fantasy football mentions found
        """
        mentions = []
        
        # Create tasks for all feeds
        tasks = []
        for feed_url in self.feeds:
            task = self._scan_feed(feed_url, keywords)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error scanning fantasy football feed: {result}")
            else:
                mentions.extend(result)
        
        # Remove duplicates and sort by relevance
        mentions = self._deduplicate_mentions(mentions)
        mentions = sorted(mentions, key=lambda x: x.relevance_score or 0, reverse=True)
        
        logger.info(f"Found {len(mentions)} fantasy football mentions")
        return mentions[:max_results]
    
    async def _scan_feed(self, feed_url: str, keywords: List[str]) -> List[MarketMention]:
        """
        Scan a single fantasy football feed
        
        Args:
            feed_url: URL of the feed
            keywords: Keywords to search for
            
        Returns:
            List of fantasy football mentions found
        """
        logger.info(f"Scanning fantasy football feed: {feed_url}")
        
        # Handle different feed types
        if "sleeper.app" in feed_url:
            return await self._scan_sleeper_api(feed_url, keywords)
        else:
            return await self._scan_rss_feed(feed_url, keywords)
    
    async def _scan_rss_feed(self, feed_url: str, keywords: List[str]) -> List[MarketMention]:
        """
        Scan a standard RSS feed
        
        Args:
            feed_url: URL of the RSS feed
            keywords: Keywords to search for
            
        Returns:
            List of fantasy football mentions found
        """
        response_data = await self._make_request(feed_url)
        if not response_data:
            return []
        
        # Parse XML content
        if isinstance(response_data, dict) and "text" in response_data:
            xml_content = response_data["text"]
        else:
            xml_content = str(response_data)
        
        return self._parse_rss_xml(xml_content, keywords, feed_url)
    
    async def _scan_sleeper_api(self, feed_url: str, keywords: List[str]) -> List[MarketMention]:
        """
        Scan Sleeper API for fantasy football news
        
        Args:
            feed_url: Sleeper API URL
            keywords: Keywords to search for
            
        Returns:
            List of fantasy football mentions found
        """
        # Sleeper API doesn't require authentication for news endpoint
        response_data = await self._make_request(feed_url)
        if not response_data:
            return []
        
        return self._parse_sleeper_response(response_data, keywords)
    
    def _parse_rss_xml(self, xml_content: str, keywords: List[str], feed_url: str) -> List[MarketMention]:
        """
        Parse RSS XML content into fantasy football mentions
        
        Args:
            xml_content: Raw XML content from RSS feed
            keywords: Keywords used in search
            feed_url: URL of the RSS feed
            
        Returns:
            List of parsed fantasy football mentions
        """
        mentions = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Handle different RSS formats
            items = []
            if root.tag.endswith("rss"):
                # Standard RSS format
                channel = root.find("channel")
                if channel is not None:
                    items = channel.findall("item")
            elif root.tag.endswith("feed"):
                # Atom format
                items = root.findall("entry")
            
            for item in items:
                try:
                    mention = self._parse_rss_item(item, keywords, feed_url)
                    if mention:
                        mentions.append(mention)
                except Exception as e:
                    logger.error(f"Error parsing RSS item: {e}")
                    continue
                    
        except ET.ParseError as e:
            logger.error(f"Error parsing RSS XML: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing RSS: {e}")
        
        return mentions
    
    def _parse_rss_item(self, item: ET.Element, keywords: List[str], feed_url: str) -> Optional[MarketMention]:
        """
        Parse a single RSS item into a fantasy football mention
        
        Args:
            item: XML element representing an RSS item
            keywords: Keywords used in search
            feed_url: URL of the RSS feed
            
        Returns:
            Fantasy football mention or None if not relevant
        """
        # Extract title and description
        title_elem = item.find("title")
        title = title_elem.text if title_elem is not None else ""
        
        desc_elem = item.find("description") or item.find("summary")
        description = desc_elem.text if desc_elem is not None else ""
        
        content = f"{title} {description}".strip()
        
        # Extract link
        link_elem = item.find("link")
        if link_elem is not None:
            url = link_elem.text or link_elem.get("href", "")
        else:
            url = ""
        
        # Extract publication date
        pub_date_elem = item.find("pubDate") or item.find("published")
        if pub_date_elem is not None:
            pub_date_str = pub_date_elem.text
            try:
                # Parse various date formats
                from dateutil import parser
                published_at = parser.parse(pub_date_str)
            except:
                published_at = datetime.now()
        else:
            published_at = datetime.now()
        
        # Calculate relevance score with fantasy football boost
        relevance_score = self._calculate_fantasy_relevance_score(content, keywords)
        
        # Only include relevant items
        if relevance_score > 0.1:  # Minimum relevance threshold
            # Extract source name from feed URL
            source = self._extract_source_name(feed_url)
            
            return MarketMention(
                source=source,
                title=title,
                content=content,
                url=url,
                published_at=published_at,
                keywords_found=self._extract_keywords(content, keywords),
                relevance_score=relevance_score
            )
        
        return None
    
    def _parse_sleeper_response(self, response_data: Dict[str, Any], keywords: List[str]) -> List[MarketMention]:
        """
        Parse Sleeper API response into fantasy football mentions
        
        Args:
            response_data: Raw response from Sleeper API
            keywords: Keywords used in search
            
        Returns:
            List of parsed fantasy football mentions
        """
        mentions = []
        
        # Sleeper API returns a list of news items
        if isinstance(response_data, list):
            for item in response_data:
                try:
                    title = item.get("title", "")
                    content = item.get("summary", "") or item.get("content", "")
                    url = item.get("url", "")
                    
                    # Parse published date
                    published_str = item.get("created_at", "")
                    if published_str:
                        try:
                            published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                        except:
                            published_at = datetime.now()
                    else:
                        published_at = datetime.now()
                    
                    # Calculate relevance score
                    full_content = f"{title} {content}".strip()
                    relevance_score = self._calculate_fantasy_relevance_score(full_content, keywords)
                    
                    if relevance_score > 0.1:
                        mention = MarketMention(
                            source="Sleeper",
                            title=title,
                            content=full_content,
                            url=url,
                            published_at=published_at,
                            keywords_found=self._extract_keywords(full_content, keywords),
                            relevance_score=relevance_score
                        )
                        mentions.append(mention)
                        
                except Exception as e:
                    logger.error(f"Error parsing Sleeper item: {e}")
                    continue
        
        return mentions
    
    def _calculate_fantasy_relevance_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate relevance score with fantasy football specific boost
        
        Args:
            text: Text to analyze
            keywords: Keywords to match against
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Start with base relevance score
        base_score = self._calculate_relevance_score(text, keywords)
        
        # Boost score for fantasy football specific terms
        text_lower = text.lower()
        fantasy_boost = 0.0
        
        for fantasy_keyword in self.fantasy_keywords:
            if fantasy_keyword.lower() in text_lower:
                fantasy_boost += 0.1
        
        # Apply fantasy football boost (capped at 0.3)
        fantasy_boost = min(fantasy_boost, 0.3)
        
        return min(base_score + fantasy_boost, 1.0)
    
    def _extract_source_name(self, feed_url: str) -> str:
        """
        Extract source name from feed URL
        
        Args:
            feed_url: URL of the feed
            
        Returns:
            Source name
        """
        parsed = urlparse(feed_url)
        domain = parsed.netloc.lower()
        
        # Map domains to readable names
        domain_map = {
            "feeds.feedburner.com": "ESPN Fantasy",
            "www.fantasypros.com": "FantasyPros",
            "www.sleeper.app": "Sleeper",
            "www.fantasyalarm.com": "Fantasy Alarm",
            "www.rotoworld.com": "RotoWorld",
            "www.numberfire.com": "NumberFire",
            "www.fantasyfootballtoday.com": "Fantasy Football Today",
            "www.thefantasyfootballers.com": "The Fantasy Footballers"
        }
        
        return domain_map.get(domain, domain)
    
    def _deduplicate_mentions(self, mentions: List[MarketMention]) -> List[MarketMention]:
        """
        Remove duplicate mentions based on URL and title similarity
        
        Args:
            mentions: List of mentions to deduplicate
            
        Returns:
            Deduplicated list of mentions
        """
        seen_urls = set()
        seen_titles = set()
        unique_mentions = []
        
        for mention in mentions:
            # Check for URL duplicates
            if mention.url in seen_urls:
                continue
            
            # Check for title duplicates (case-insensitive)
            title_lower = mention.title.lower()
            if title_lower in seen_titles:
                continue
            
            seen_urls.add(mention.url)
            seen_titles.add(title_lower)
            unique_mentions.append(mention)
        
        return unique_mentions
