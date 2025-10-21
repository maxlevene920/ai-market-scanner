"""
News API scanner for market data collection

This module implements a scanner for the News API service,
providing access to news articles from various sources.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger

from .base_scanner import BaseScanner, MarketMention
from ..config import settings


class NewsScanner(BaseScanner):
    """
    Scanner for News API
    
    This scanner fetches news articles from the News API service
    and filters them for market-related content.
    """
    
    def __init__(self):
        """Initialize the News API scanner"""
        super().__init__("News API", rate_limit=1.0)
        self.api_key = settings.api.news_api_key
        self.base_url = "https://newsapi.org/v2/everything"
        
        if not self.api_key:
            logger.warning("News API key not configured")
    
    async def scan(self, keywords: List[str], max_results: int = 100) -> List[MarketMention]:
        """
        Scan News API for market mentions
        
        Args:
            keywords: Keywords to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of market mentions found
        """
        if not self.api_key:
            logger.error("News API key not configured")
            return []
        
        mentions = []
        
        # Search for each keyword combination
        for keyword in keywords:
            if len(mentions) >= max_results:
                break
            
            query_params = self._build_query_params(keyword, max_results - len(mentions))
            url = f"{self.base_url}?{query_params}"
            
            logger.info(f"Searching News API for: {keyword}")
            
            response_data = await self._make_request(url)
            if response_data:
                keyword_mentions = self._parse_response(response_data, [keyword])
                mentions.extend(keyword_mentions)
                
                # Rate limiting between requests
                await asyncio.sleep(1.0)
        
        # Remove duplicates and sort by relevance
        mentions = self._deduplicate_mentions(mentions)
        mentions = sorted(mentions, key=lambda x: x.relevance_score or 0, reverse=True)
        
        logger.info(f"Found {len(mentions)} market mentions from News API")
        return mentions[:max_results]
    
    def _build_query_params(self, keyword: str, page_size: int) -> str:
        """
        Build query parameters for News API request
        
        Args:
            keyword: Search keyword
            page_size: Number of results to request
            
        Returns:
            URL query string
        """
        from urllib.parse import urlencode
        
        # Calculate date range (last 7 days)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)
        
        params = {
            "q": keyword,
            "apiKey": self.api_key,
            "pageSize": min(page_size, 100),  # News API max is 100
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
            "language": "en",
            "domains": "reuters.com,bloomberg.com,cnbc.com,marketwatch.com,wsj.com,ft.com"
        }
        
        return urlencode(params)
    
    def _parse_response(self, response_data: Dict[str, Any], keywords: List[str]) -> List[MarketMention]:
        """
        Parse News API response into market mentions
        
        Args:
            response_data: Raw response from News API
            keywords: Keywords used in search
            
        Returns:
            List of parsed market mentions
        """
        mentions = []
        
        if response_data.get("status") != "ok":
            logger.error(f"News API error: {response_data.get('message', 'Unknown error')}")
            return mentions
        
        articles = response_data.get("articles", [])
        
        for article in articles:
            try:
                # Extract article data
                title = article.get("title", "")
                description = article.get("description", "")
                content = f"{title} {description}".strip()
                url = article.get("url", "")
                source = article.get("source", {}).get("name", "News API")
                
                # Parse published date
                published_str = article.get("publishedAt", "")
                if published_str:
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                else:
                    published_at = datetime.now()
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(content, keywords)
                
                # Only include relevant articles
                if relevance_score > 0.1:  # Minimum relevance threshold
                    mention = MarketMention(
                        source=source,
                        title=title,
                        content=content,
                        url=url,
                        published_at=published_at,
                        keywords_found=self._extract_keywords(content, keywords),
                        relevance_score=relevance_score
                    )
                    mentions.append(mention)
                    
            except Exception as e:
                logger.error(f"Error parsing article: {e}")
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
