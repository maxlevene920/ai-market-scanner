"""
Base scanner class for market data collection

This module provides the abstract base class that all market scanners inherit from.
It defines the common interface and shared functionality for scanning operations.
"""

import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from loguru import logger


@dataclass
class MarketMention:
    """Data structure for a market mention"""
    source: str
    title: str
    content: str
    url: str
    published_at: datetime
    sentiment_score: Optional[float] = None
    keywords_found: List[str] = None
    relevance_score: Optional[float] = None
    
    def __post_init__(self):
        """Initialize default values"""
        if self.keywords_found is None:
            self.keywords_found = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "sentiment_score": self.sentiment_score,
            "keywords_found": self.keywords_found,
            "relevance_score": self.relevance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketMention':
        """Create instance from dictionary"""
        return cls(
            source=data["source"],
            title=data["title"],
            content=data["content"],
            url=data["url"],
            published_at=datetime.fromisoformat(data["published_at"]),
            sentiment_score=data.get("sentiment_score"),
            keywords_found=data.get("keywords_found", []),
            relevance_score=data.get("relevance_score")
        )


class BaseScanner(ABC):
    """
    Abstract base class for market scanners
    
    This class defines the common interface and shared functionality
    that all market scanners must implement.
    """
    
    def __init__(self, name: str, rate_limit: float = 1.0):
        """
        Initialize the base scanner
        
        Args:
            name: Name of the scanner
            rate_limit: Maximum requests per second
        """
        self.name = name
        self.rate_limit = rate_limit
        self.session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        logger.info(f"Initialized {self.name} scanner with rate limit: {rate_limit} req/s")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
    
    async def start(self):
        """Start the scanner session"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=30)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "AI-Market-Scanner/1.0"}
        )
        
        # Create semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(int(self.rate_limit))
        
        logger.info(f"Started {self.name} scanner session")
    
    async def stop(self):
        """Stop the scanner session"""
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info(f"Stopped {self.name} scanner session")
    
    async def _rate_limit_request(self):
        """Apply rate limiting to requests"""
        if self._semaphore:
            await self._semaphore.acquire()
            # Release after delay to maintain rate limit
            asyncio.create_task(self._release_after_delay())
    
    async def _release_after_delay(self):
        """Release semaphore after delay"""
        await asyncio.sleep(1.0 / self.rate_limit)
        if self._semaphore:
            self._semaphore.release()
    
    async def _make_request(self, url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """
        Make an HTTP request with error handling and rate limiting
        
        Args:
            url: URL to request
            headers: Optional headers to include
            
        Returns:
            Response data as dictionary or None if failed
        """
        if not self.session:
            logger.error(f"Scanner {self.name} session not started")
            return None
        
        await self._rate_limit_request()
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    
                    if 'application/json' in content_type:
                        return await response.json()
                    else:
                        text = await response.text()
                        return {"text": text, "content_type": content_type}
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout requesting {url}")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"Client error requesting {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error requesting {url}: {e}")
            return None
    
    def _extract_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """
        Extract matching keywords from text
        
        Args:
            text: Text to search in
            keywords: Keywords to search for
            
        Returns:
            List of found keywords
        """
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _calculate_relevance_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate relevance score based on keyword matches
        
        Args:
            text: Text to analyze
            keywords: Keywords to match against
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not keywords:
            return 0.0
        
        found_keywords = self._extract_keywords(text, keywords)
        
        # Base score from keyword matches
        keyword_score = len(found_keywords) / len(keywords)
        
        # Bonus for multiple occurrences
        text_lower = text.lower()
        total_occurrences = sum(text_lower.count(keyword.lower()) for keyword in found_keywords)
        occurrence_bonus = min(total_occurrences * 0.1, 0.3)
        
        return min(keyword_score + occurrence_bonus, 1.0)
    
    @abstractmethod
    async def scan(self, keywords: List[str], max_results: int = 100) -> List[MarketMention]:
        """
        Scan for market mentions
        
        Args:
            keywords: Keywords to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of market mentions found
        """
        pass
    
    @abstractmethod
    def _parse_response(self, response_data: Dict[str, Any], keywords: List[str]) -> List[MarketMention]:
        """
        Parse response data into market mentions
        
        Args:
            response_data: Raw response data
            keywords: Keywords used in search
            
        Returns:
            List of parsed market mentions
        """
        pass
