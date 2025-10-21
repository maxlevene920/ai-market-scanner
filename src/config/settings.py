"""
Configuration management for the AI Market Scanner

This module handles all configuration settings including API keys,
market sources, and scanning parameters.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class APIConfig:
    """Configuration for external APIs"""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    news_api_key: Optional[str] = None
    twitter_api_key: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    
    def __post_init__(self):
        """Load API keys from environment variables"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.twitter_api_key = os.getenv("TWITTER_API_KEY")
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")


@dataclass
class MarketSource:
    """Configuration for a market data source"""
    name: str
    url: str
    enabled: bool = True
    rate_limit: float = 1.0  # requests per second
    headers: Optional[Dict[str, str]] = None
    requires_auth: bool = False


@dataclass
class ScannerConfig:
    """Configuration for the market scanner"""
    scan_interval_minutes: int = 30
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    cache_duration_hours: int = 24
    
    # Keywords and filters
    keywords: List[str] = None
    exclude_keywords: List[str] = None
    min_mention_threshold: int芙 5
    
    def __post_init__(self):
        """Set default keywords if none provided"""
        if self.keywords is None:
            self.keywords = [
                "market", "trading", "investment", "finance", "crypto",
                "stocks", "forex", "commodities", "bonds", "derivatives"
            ]
        
        if self.exclude_keywords is None:
            self.exclude_keywords = [
                "spam", "scam", "fake", "clickbait"
            ]


@dataclass
class AgentConfig:
    """Configuration for the AI agent"""
    model_provider: str = "openai"  # "openai" or "anthropic"
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    analysis_depth: str = "medium"  # "shallow", "medium", "deep"
    
    # Prompt templates
    analysis_prompt_template: str = """
    Analyze the following market mentions and provide insights:
    
    Data: {data}
    
    Please provide:
    1. Key trends and patterns
    2. Market sentiment analysis
    3. Potential opportunities
    4. Risk factors
    5. Recommendations
    
    Format your response in a structured manner.
    """


class Settings:
    """Main settings class that aggregates all configurations"""
    
    def __init__(self):
        self.api = APIConfig()
        self.scanner = ScannerConfig()
        self.agent = AgentConfig()
        self.market_sources = self._load_market_sources()
    
    def _load_market_sources(self) -> List[MarketSource]:
        """Load configured market sources"""
        return [
            MarketSource(
                name="News API",
                url="https://newsapi.org/v2/everything",
                requires_auth=True,
                headers={"X-API-Key": self.api.news_api_key} if self.api.news_api_key else None
            ),
            MarketSource(
                name="Reddit",
                url="https://www.reddit.com/r/investing/new.json",
                rate_limit=0.5
            ),
            MarketSource(
                name="Financial News",
                url="https://feeds.finance.yahoo.com/rss/2.0/headline",
                rate_limit=2.0
            ),
            MarketSource(
                name="Crypto News",
                url="https://cointelegraph.com/rss",
                rate_limit=1.0
            )
        ]
    
    def get_enabled_sources(self) -> List[MarketSource]:
        """Get only enabled market sources"""
        return [source for source in self.market_sources if source.enabled]
    
    def validate_config(self) -> bool:
        """Validate that all required configurations are present"""
        errors = []
        
        # Check for at least one AI provider
        if not self.api.openai_api_key and not self.api.anthropic_api_key:
            errors.append("At least one AI provider API key is required")
        
        # Check for enabled sources
        if not self.get_enabled_sources():
            errors.append("At least one market source must be enabled")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# Global settings instance
settings = Settings()
