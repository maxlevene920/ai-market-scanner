"""
RSS scanner for market data collection

This module implements a scanner for RSS feeds from various
financial news sources and market data providers.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse

from .base_scanner import BaseScanner, MarketMention


class RSSScanner(BaseScanner):
    """
    Scanner for RSS feeds
    
    This scanner fetches and parses RSS feeds from various
    financial news sources and market data providers.
    """
    
    def __init__(self):
        """Initialize the RSS scanner"""
        super().__init__("RSS", rate_limit=2.0)
        self.feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://feeds.marketwatch.com/marketwatch/topstories/",
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://feeds.feedburner.com/oreilly/radar"
        ]
    
    async def scan(self, keywords: List[str], max_results: int = 100) -> List[MarketMention]:
        """
        Scan RSS feeds for market mentions
        
        Args:
            keywords: Keywords to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of market mentions found
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
                logger.error(f"Error scanning RSS feed: {result}")
            else:
                mentions.extend(result)
        
        # Remove duplicates and sort by relevance
        mentions = self._deduplicate_mentions(mentions)
        mentions = sorted(mentions, key=lambda x: x.relevance_score or 0, reverse=True)
        
        logger.info(f"Found {len(mentions)} market mentions from RSS feeds")
        return mentions[:max_results]
    
    async def _scan_feed(self, feed_url: str, keywords: List[str]) -> List[MarketMention]:
        """
        Scan a single RSS feed
        
        Args:
            feed_url: URL of the RSS feed
            keywords: Keywords to search for
            
        Returns:
            List of market mentions found
        """
        logger.info(f"Scanning RSS feed: {feed_url}")
        
        response_data = await self._make_request(feed_url)
        if not response_data:
            return []
        
        # Parse XML content
        if isinstance(response_data, dict) and "text" in response_data:
            xml_content = response_data["text"]
        else:
            xml_content = str(response_data)
        
        return self._parse_rss_xml(xml_content, keywords, feed_url)
    
    def _parse_rss_xml(self, xml_content: str, keywords: List[str], feed_url: str) -> List[MarketMention]:
        """
        Parse RSS XML content into market mentions
        
        Args:
            xml_content: Raw XML content from RSS feed
            keywords: Keywords used in search
            feed_url: URL of the RSS feed
            
        Returns:
            List of parsed market mentions
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
        Parse a single RSS item into a market mention
        
        Args:
            item: XML element representing an RSS item
            keywords: Keywords used in search
            feed_url: URL of the RSS feed
            
        Returns:
            Market mention or None if not relevant
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
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(content, keywords)
        
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
    
    def _extract_source_name(self, feed_url: str) -> str:
        """
        Extract source name from feed URL
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            Source name
        """
        parsed = urlparse(feed_url)
        domain = parsed.netloc.lower()
        
        # Map domains to readable names
        domain_map = {
            "feeds.finance.yahoo.com": "Yahoo Finance",
            "feeds.reuters.com": "Reuters",
            "feeds.bloomberg.com": "Bloomberg",
            "www.cnbc.com": "CNBC",
            "feeds.marketwatch.com": "MarketWatch",
            "cointelegraph.com": "CoinTelegraph",
            "www.coindesk.com": "CoinDesk",
            "feeds.feedburner.com": "FeedBurner"
        }
        
        return domain_map.get(domain, domain)
    
    def _deduplicate_mentions(self, mentions: List[MarketMention]) -> List[MarketMention]:
        """
        Remove duplicate mentions based on URL
        
        Args:
            mentions: List of mentions to deduplicate
            
        Returns:
            Deduplicated list of mentions
        """
        seen_urls = set()
        unique_mentions = []
        
        for mention in mentions:
            if mention.url not in seen_urls:
                seen_urls.add(mention.url)
                unique_mentions.append(mention)
        
        return unique_mentions
