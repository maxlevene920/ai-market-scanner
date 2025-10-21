"""
Main market scanner orchestrator

This module coordinates multiple scanners to collect market mentions
from various sources and provides a unified interface for scanning operations.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .base_scanner import MarketMention
from .news_scanner import NewsScanner
from .reddit_scanner import RedditScanner
from .rss_scanner import RSSScanner
from ..config import settings


class MarketScanner:
    """
    Main market scanner that orchestrates multiple data sources
    
    This class coordinates scanning operations across multiple sources
    and provides a unified interface for market data collection.
    """
    
    def __init__(self):
        """Initialize the market scanner"""
        self.scanners = []
        self._initialize_scanners()
        
        logger.info(f"Initialized MarketScanner with {len(self.scanners)} scanners")
    
    def _initialize_scanners(self):
        """Initialize available scanners based on configuration"""
        # Always initialize RSS scanner (no API key required)
        self.scanners.append(RSSScanner())
        
        # Initialize Reddit scanner (no API key required)
        self.scanners.append(RedditScanner())
        
        # Initialize News API scanner if API key is available
        if settings.api.news_api_key:
            self.scanners.append(NewsScanner())
        else:
            logger.warning("News API scanner not initialized - API key missing")
    
    async def scan_markets(self, keywords: Optional[List[str]] = None, max_results: int = 200) -> List[MarketMention]:
        """
        Scan all available sources for market mentions
        
        Args:
            keywords: Keywords to search for (uses config defaults if None)
            max_results: Maximum number of results to return
            
        Returns:
            List of market mentions from all sources
        """
        if keywords is None:
            keywords = settings.scanner.keywords
        
        logger.info(f"Starting market scan with keywords: {keywords}")
        logger.info(f"Target results: {max_results}")
        
        all_mentions = []
        
        # Create tasks for all scanners
        tasks = []
        for scanner in self.scanners:
            task = self._scan_with_scanner(scanner, keywords, max_results)
            tasks.append(task)
        
        # Execute all scanner tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results from all scanners
        for i, result in enumerate(results):
            scanner_name = self.scanners[i].name
            if isinstance(result, Exception):
                logger.error(f"Scanner {scanner_name} failed: {result}")
            else:
                logger.info(f"Scanner {scanner_name} found {len(result)} mentions")
                all_mentions.extend(result)
        
        # Process and filter mentions
        processed_mentions = self._process_mentions(all_mentions, keywords)
        
        logger.info(f"Total mentions found: {len(all_mentions)}")
        logger.info(f"Relevant mentions after filtering: {len(processed_mentions)}")
        
        return processed_mentions[:max_results]
    
    async def _scan_with_scanner(self, scanner: Any, keywords: List[str], max_results: int) -> List[MarketMention]:
        """
        Scan with a specific scanner
        
        Args:
            scanner: Scanner instance to use
            keywords: Keywords to search for
            max_results: Maximum results per scanner
            
        Returns:
            List of mentions from the scanner
        """
        try:
            async with scanner:
                # Distribute max_results across scanners
                scanner_max = max_results // len(self.scanners)
                return await scanner.scan(keywords, scanner_max)
        except Exception as e:
            logger.error(f"Error with scanner {scanner.name}: {e}")
            return []
    
    def _process_mentions(self, mentions: List[MarketMention], keywords: List[str]) -> List[MarketMention]:
        """
        Process and filter mentions based on relevance and quality
        
        Args:
            mentions: Raw mentions from all scanners
            keywords: Keywords used in search
            
        Returns:
            Processed and filtered mentions
        """
        # Remove duplicates based on URL
        unique_mentions = self._deduplicate_mentions(mentions)
        
        # Filter by relevance score
        relevant_mentions = [
            mention for mention in unique_mentions
            if mention.relevance_score and mention.relevance_score >= 0.2
        ]
        
        # Filter out excluded keywords
        filtered_mentions = self._filter_excluded_keywords(relevant_mentions)
        
        # Sort by relevance score and publication date
        sorted_mentions = sorted(
            filtered_mentions,
            key=lambda x: (x.relevance_score or 0, x.published_at),
            reverse=True
        )
        
        return sorted_mentions
    
    def _deduplicate_mentions(self, mentions: List[MarketMention]) -> List[MarketMention]:
        """
        Remove duplicate mentions based on URL and content similarity
        
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
    
    def _filter_excluded_keywords(self, mentions: List[MarketMention]) -> List[MarketMention]:
        """
        Filter out mentions containing excluded keywords
        
        Args:
            mentions: List of mentions to filter
            
        Returns:
            Filtered list of mentions
        """
        exclude_keywords = settings.scanner.exclude_keywords
        
        if not exclude_keywords:
            return mentions
        
        filtered_mentions = []
        
        for mention in mentions:
            content_lower = mention.content.lower()
            
            # Check if any excluded keyword is present
            has_excluded = any(
                keyword.lower() in content_lower
                for keyword in exclude_keywords
            )
            
            if not has_excluded:
                filtered_mentions.append(mention)
        
        return filtered_mentions
    
    def get_scanner_status(self) -> Dict[str, Any]:
        """
        Get status information about all scanners
        
        Returns:
            Dictionary with scanner status information
        """
        status = {
            "total_scanners": len(self.scanners),
            "scanners": []
        }
        
        for scanner in self.scanners:
            scanner_info = {
                "name": scanner.name,
                "rate_limit": scanner.rate_limit,
                "enabled": True
            }
            status["scanners"].append(scanner_info)
        
        return status
    
    async def test_scanners(self) -> Dict[str, bool]:
        """
        Test all scanners to ensure they're working properly
        
        Returns:
            Dictionary mapping scanner names to test results
        """
        test_results = {}
        
        for scanner in self.scanners:
            try:
                async with scanner:
                    # Test with a simple keyword
                    test_mentions = await scanner.scan(["test"], max_results=1)
                    test_results[scanner.name] = True
                    logger.info(f"Scanner {scanner.name} test passed")
            except Exception as e:
                test_results[scanner.name] = False
                logger.error(f"Scanner {scanner.name} test failed: {e}")
        
        return test_results
