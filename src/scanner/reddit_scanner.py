"""
Reddit scanner for market data collection

This module implements a scanner for Reddit posts and comments,
focusing on investment and market-related subreddits.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .base_scanner import BaseScanner, MarketMention
from ..config import settings


class RedditScanner(BaseScanner):
    """
    Scanner for Reddit posts and comments
    
    This scanner fetches posts from investment and market-related
    subreddits and analyzes them for market mentions.
    """
    
    def __init__(self):
        """Initialize the Reddit scanner"""
        super().__init__("Reddit", rate_limit=0.5)  # Reddit has strict rate limits
        self.base_url = "https://www.reddit.com"
        self.subreddits = [
            "investing", "stocks", "SecurityAnalysis", "ValueInvesting",
            "cryptocurrency", "CryptoCurrency", "Bitcoin", "ethereum",
            "wallstreetbets", "options", "investing", "personalfinance"
        ]
    
    async def scan(self, keywords: List[str], max_results: int = 100) -> List[MarketMention]:
        """
        Scan Reddit for market mentions
        
        Args:
            keywords: Keywords to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of market mentions found
        """
        mentions = []
        
        # Search each subreddit
        for subreddit in self.subreddits:
            if len(mentions) >= max_results:
                break
            
            logger.info(f"Scanning r/{subreddit} for market mentions")
            
            # Get hot posts
            hot_mentions = await self._scan_subreddit(subreddit, "hot", keywords)
            mentions.extend(hot_mentions)
            
            # Get new posts
            new_mentions = await self._scan_subreddit(subreddit, "new", keywords)
            mentions.extend(new_mentions)
            
            # Rate limiting between subreddits
            await asyncio.sleep(2.0)
        
        # Remove duplicates and sort by relevance
        mentions = self._deduplicate_mentions(mentions)
        mentions = sorted(mentions, key=lambda x: x.relevance_score or 0, reverse=True)
        
        logger.info(f"Found {len(mentions)} market mentions from Reddit")
        return mentions[:max_results]
    
    async def _scan_subreddit(self, subreddit: str, sort: str, keywords: List[str]) -> List[MarketMention]:
        """
        Scan a specific subreddit for mentions
        
        Args:
            subreddit: Name of the subreddit
            sort: Sort order (hot, new, top)
            keywords: Keywords to search for
            
        Returns:
            List of market mentions found
        """
        url = f"{self.base_url}/r/{subreddit}/{sort}.json?limit=25"
        
        response_data = await self._make_request(url)
        if not response_data:
            return []
        
        return self._parse_response(response_data, keywords, subreddit)
    
    def _parse_response(self, response_data: Dict[str, Any], keywords: List[str], subreddit: str) -> List[MarketMention]:
        """
        Parse Reddit JSON response into market mentions
        
        Args:
            response_data: Raw response from Reddit API
            keywords: Keywords used in search
            subreddit: Name of the subreddit
            
        Returns:
            List of parsed market mentions
        """
        mentions = []
        
        data = response_data.get("data", {})
        posts = data.get("children", [])
        
        for post_data in posts:
            try:
                post = post_data.get("data", {})
                
                # Extract post data
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                content = f"{title} {selftext}".strip()
                url = f"{self.base_url}{post.get('permalink', '')}"
                
                # Parse created date
                created_utc = post.get("created_utc", 0)
                published_at = datetime.fromtimestamp(created_utc)
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(content, keywords)
                
                # Only include relevant posts
                if relevance_score > 0.1:  # Minimum relevance threshold
                    mention = MarketMention(
                        source=f"r/{subreddit}",
                        title=title,
                        content=content,
                        url=url,
                        published_at=published_at,
                        keywords_found=self._extract_keywords(content, keywords),
                        relevance_score=relevance_score
                    )
                    mentions.append(mention)
                    
            except Exception as e:
                logger.error(f"Error parsing Reddit post: {e}")
                continue
        
        return mentions
    
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
