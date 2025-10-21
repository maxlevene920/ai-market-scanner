"""
Base AI agent class for market analysis

This module provides the abstract base class that all AI agents inherit from.
It defines the common interface and shared functionality for AI-powered analysis.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
from loguru import logger

from ..scanner.base_scanner import MarketMention
from ..config import settings


@dataclass
class AnalysisResult:
    """Data structure for analysis results"""
    analysis_type: str
    timestamp: datetime
    data: Dict[str, Any]
    confidence_score: Optional[float] = None
    insights: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        """Initialize default values"""
        if self.insights is None:
            self.insights = []
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "analysis_type": self.analysis_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "confidence_score": self.confidence_score,
            "insights": self.insights,
            "recommendations": self.recommendations
        }


class BaseAgent(ABC):
    """
    Abstract base class for AI agents
    
    This class defines the common interface and shared functionality
    that all AI agents must implement for market analysis.
    """
    
    def __init__(self, name: str, model_provider: str = None):
        """
        Initialize the base agent
        
        Args:
            name: Name of the agent
            model_provider: AI model provider to use
        """
        self.name = name
        self.model_provider = model_provider or settings.agent.model_provider
        self.model_name = settings.agent.model_name
        self.temperature = settings.agent.temperature
        self.max_tokens = settings.agent.max_tokens
        
        logger.info(f"Initialized {self.name} agent with {self.model_provider}")
    
    @abstractmethod
    async def analyze(self, mentions: List[MarketMention]) -> AnalysisResult:
        """
        Analyze market mentions
        
        Args:
            mentions: List of market mentions to analyze
            
        Returns:
            Analysis result containing insights and recommendations
        """
        pass
    
    def _prepare_data_for_analysis(self, mentions: List[MarketMention]) -> Dict[str, Any]:
        """
        Prepare market mentions data for AI analysis
        
        Args:
            mentions: List of market mentions
            
        Returns:
            Structured data for analysis
        """
        # Group mentions by source
        sources = {}
        for mention in mentions:
            source = mention.source
            if source not in sources:
                sources[source] = []
            sources[source].append(mention)
        
        # Calculate basic statistics
        total_mentions = len(mentions)
        avg_relevance = sum(m.relevance_score or 0 for m in mentions) / total_mentions if total_mentions > 0 else 0
        
        # Extract keywords frequency
        keyword_freq = {}
        for mention in mentions:
            for keyword in mention.keywords_found:
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # Sort keywords by frequency
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_mentions": total_mentions,
            "sources": {source: len(mentions) for source, mentions in sources.items()},
            "average_relevance": avg_relevance,
            "top_keywords": top_keywords,
            "time_range": {
                "earliest": min(m.published_at for m in mentions).isoformat() if mentions else None,
                "latest": max(m.published_at for m in mentions).isoformat() if mentions else None
            },
            "sample_content": [m.content[:200] for m in mentions[:5]]  # First 200 chars of first 5 mentions
        }
    
    def _format_mentions_for_prompt(self, mentions: List[MarketMention], max_mentions: int = 20) -> str:
        """
        Format mentions for inclusion in AI prompts
        
        Args:
            mentions: List of market mentions
            max_mentions: Maximum number of mentions to include
            
        Returns:
            Formatted string of mentions
        """
        if not mentions:
            return "No market mentions available for analysis."
        
        # Take most relevant mentions
        sorted_mentions = sorted(mentions, key=lambda x: x.relevance_score or 0, reverse=True)
        selected_mentions = sorted_mentions[:max_mentions]
        
        formatted = []
        for i, mention in enumerate(selected_mentions, 1):
            formatted.append(
                f"{i}. Source: {mention.source}\n"
                f"   Title: {mention.title}\n"
                f"   Content: {mention.content[:300]}...\n"
                f"   Published: {mention.published_at.strftime('%Y-%m-%d %H:%M')}\n"
                f"   Relevance: {mention.relevance_score:.2f}\n"
                f"   Keywords: {', '.join(mention.keywords_found)}\n"
            )
        
        return "\n".join(formatted)
    
    def _extract_insights_from_response(self, response: str) -> List[str]:
        """
        Extract insights from AI response text
        
        Args:
            response: Raw response from AI model
            
        Returns:
            List of extracted insights
        """
        insights = []
        
        # Look for numbered lists or bullet points
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('•', '-', '*', '1.', '2.', '3.')) or 
                        'insight' in line.lower() or 'finding' in line.lower()):
                # Clean up the line
                cleaned = line.lstrip('•-*123456789. ')
                if len(cleaned) > 10:  # Only include substantial insights
                    insights.append(cleaned)
        
        return insights[:10]  # Limit to top 10 insights
    
    def _extract_recommendations_from_response(self, response: str) -> List[str]:
        """
        Extract recommendations from AI response text
        
        Args:
            response: Raw response from AI model
            
        Returns:
            List of extracted recommendations
        """
        recommendations = []
        
        # Look for recommendation keywords
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and any(keyword in line.lower() for keyword in 
                          ['recommend', 'suggest', 'advise', 'consider', 'should', 'action']):
                # Clean up the line
                cleaned = line.lstrip('•-*123456789. ')
                if len(cleaned) > 10:  # Only include substantial recommendations
                    recommendations.append(cleaned)
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_confidence_score(self, mentions: List[MarketMention], response_length: int) -> float:
        """
        Calculate confidence score for analysis
        
        Args:
            mentions: List of analyzed mentions
            response_length: Length of AI response
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not mentions:
            return 0.0
        
        # Base confidence on data quality
        avg_relevance = sum(m.relevance_score or 0 for m in mentions) / len(mentions)
        data_quality_score = min(avg_relevance, 1.0)
        
        # Bonus for sufficient data
        data_quantity_score = min(len(mentions) / 50, 1.0)  # Normalize to 50 mentions
        
        # Bonus for detailed response
        response_quality_score = min(response_length / 1000, 1.0)  # Normalize to 1000 chars
        
        # Weighted combination
        confidence = (
            data_quality_score * 0.5 +
            data_quantity_score * 0.3 +
            response_quality_score * 0.2
        )
        
        return min(confidence, 1.0)
