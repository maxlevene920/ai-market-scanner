"""
Trend analysis agent for market data

This module implements an AI agent that analyzes trends
in market mentions and provides trend-based insights.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from loguru import logger

from .base_agent import BaseAgent, AnalysisResult
from ..scanner.base_scanner import MarketMention
from ..config import settings


class TrendAnalyzer(BaseAgent):
    """
    AI agent for trend analysis of market mentions
    
    This agent analyzes trends in market mentions and provides
    insights about emerging patterns, topic evolution, and market shifts.
    """
    
    def __init__(self):
        """Initialize the trend analyzer"""
        super().__init__("Trend Analyzer")
        self.trend_prompt_template = """
        Analyze the trends in the following market mentions and provide insights:
        
        Market Mentions:
        {mentions}
        
        Please provide:
        1. Emerging trends and patterns
        2. Topic evolution over time
        3. Keyword frequency trends
        4. Source-specific trends
        5. Market shift indicators
        6. Future trend predictions
        
        Format your response in a structured manner with clear insights and recommendations.
        """
    
    async def analyze(self, mentions: List[MarketMention]) -> AnalysisResult:
        """
        Analyze trends in market mentions
        
        Args:
            mentions: List of market mentions to analyze
            
        Returns:
            Analysis result containing trend insights
        """
        if not mentions:
            return AnalysisResult(
                analysis_type="trend",
                timestamp=datetime.now(),
                data={"error": "No mentions provided"},
                confidence_score=0.0,
                insights=["No data available for trend analysis"],
                recommendations=["Collect more market mentions for analysis"]
            )
        
        logger.info(f"Analyzing trends for {len(mentions)} mentions")
        
        # Prepare data for analysis
        analysis_data = self._prepare_data_for_analysis(mentions)
        formatted_mentions = self._format_mentions_for_prompt(mentions)
        
        # Calculate trend metrics
        trend_metrics = self._calculate_trend_metrics(mentions)
        analysis_data.update(trend_metrics)
        
        # Generate AI analysis
        try:
            response = await self._generate_analysis(formatted_mentions)
            
            # Extract insights and recommendations
            insights = self._extract_insights_from_response(response)
            recommendations = self._extract_recommendations_from_response(response)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(mentions, len(response))
            
            # Combine analysis data
            analysis_data.update({
                "ai_response": response,
                "total_analyzed": len(mentions)
            })
            
            result = AnalysisResult(
                analysis_type="trend",
                timestamp=datetime.now(),
                data=analysis_data,
                confidence_score=confidence_score,
                insights=insights,
                recommendations=recommendations
            )
            
            logger.info(f"Trend analysis completed with confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return AnalysisResult(
                analysis_type="trend",
                timestamp=datetime.now(),
                data={"error": str(e)},
                confidence_score=0.0,
                insights=["Trend analysis failed"],
                recommendations=["Check AI provider configuration"]
            )
    
    def _calculate_trend_metrics(self, mentions: List[MarketMention]) -> Dict[str, Any]:
        """
        Calculate trend metrics from mentions
        
        Args:
            mentions: List of market mentions
            
        Returns:
            Dictionary with trend metrics
        """
        # Sort mentions by publication date
        sorted_mentions = sorted(mentions, key=lambda x: x.published_at)
        
        # Time-based analysis
        time_metrics = self._analyze_time_trends(sorted_mentions)
        
        # Keyword trend analysis
        keyword_metrics = self._analyze_keyword_trends(sorted_mentions)
        
        # Source trend analysis
        source_metrics = self._analyze_source_trends(sorted_mentions)
        
        # Volume trend analysis
        volume_metrics = self._analyze_volume_trends(sorted_mentions)
        
        return {
            "time_trends": time_metrics,
            "keyword_trends": keyword_metrics,
            "source_trends": source_metrics,
            "volume_trends": volume_metrics
        }
    
    def _analyze_time_trends(self, mentions: List[MarketMention]) -> Dict[str, Any]:
        """
        Analyze trends over time
        
        Args:
            mentions: Sorted list of mentions by time
            
        Returns:
            Time-based trend metrics
        """
        if len(mentions) < 2:
            return {"trend": "insufficient_data"}
        
        # Group mentions by hour
        hourly_counts = defaultdict(int)
        for mention in mentions:
            hour_key = mention.published_at.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1
        
        # Calculate trend direction
        hours = sorted(hourly_counts.keys())
        if len(hours) < 2:
            return {"trend": "insufficient_data"}
        
        recent_avg = sum(hourly_counts[h] for h in hours[-3:]) / min(3, len(hours))
        early_avg = sum(hourly_counts[h] for h in hours[:3]) / min(3, len(hours))
        
        trend_direction = "increasing" if recent_avg > early_avg else "decreasing" if recent_avg < early_avg else "stable"
        
        return {
            "trend": trend_direction,
            "recent_avg": recent_avg,
            "early_avg": early_avg,
            "total_hours": len(hours),
            "peak_hour": max(hourly_counts.items(), key=lambda x: x[1])[0].isoformat() if hourly_counts else None
        }
    
    def _analyze_keyword_trends(self, mentions: List[MarketMention]) -> Dict[str, Any]:
        """
        Analyze keyword trends over time
        
        Args:
            mentions: Sorted list of mentions by time
            
        Returns:
            Keyword trend metrics
        """
        # Split mentions into early and recent halves
        mid_point = len(mentions) // 2
        early_mentions = mentions[:mid_point]
        recent_mentions = mentions[mid_point:]
        
        # Count keywords in each period
        early_keywords = Counter()
        recent_keywords = Counter()
        
        for mention in early_mentions:
            early_keywords.update(mention.keywords_found)
        
        for mention in recent_mentions:
            recent_keywords.update(mention.keywords_found)
        
        # Find trending keywords
        trending_up = []
        trending_down = []
        
        all_keywords = set(early_keywords.keys()) | set(recent_keywords.keys())
        
        for keyword in all_keywords:
            early_count = early_keywords[keyword]
            recent_count = recent_keywords[keyword]
            
            if early_count > 0 and recent_count > 0:
                change_ratio = recent_count / early_count
                if change_ratio > 1.5:
                    trending_up.append((keyword, change_ratio))
                elif change_ratio < 0.67:
                    trending_down.append((keyword, change_ratio))
        
        # Sort by change ratio
        trending_up.sort(key=lambda x: x[1], reverse=True)
        trending_down.sort(key=lambda x: x[1])
        
        return {
            "trending_up": trending_up[:10],
            "trending_down": trending_down[:10],
            "total_keywords": len(all_keywords),
            "early_period_keywords": len(early_keywords),
            "recent_period_keywords": len(recent_keywords)
        }
    
    def _analyze_source_trends(self, mentions: List[MarketMention]) -> Dict[str, Any]:
        """
        Analyze trends by source
        
        Args:
            mentions: Sorted list of mentions by time
            
        Returns:
            Source trend metrics
        """
        # Group mentions by source
        source_mentions = defaultdict(list)
        for mention in mentions:
            source_mentions[mention.source].append(mention)
        
        # Calculate trends for each source
        source_trends = {}
        for source, source_mentions_list in source_mentions.items():
            if len(source_mentions_list) < 2:
                continue
            
            # Split into early and recent
            mid_point = len(source_mentions_list) // 2
            early_count = mid_point
            recent_count = len(source_mentions_list) - mid_point
            
            trend = "increasing" if recent_count > early_count else "decreasing" if recent_count < early_count else "stable"
            
            source_trends[source] = {
                "trend": trend,
                "total_mentions": len(source_mentions_list),
                "early_count": early_count,
                "recent_count": recent_count,
                "avg_relevance": sum(m.relevance_score or 0 for m in source_mentions_list) / len(source_mentions_list)
            }
        
        return {
            "source_trends": source_trends,
            "most_active_source": max(source_mentions.keys(), key=lambda x: len(source_mentions[x])) if source_mentions else None,
            "total_sources": len(source_mentions)
        }
    
    def _analyze_volume_trends(self, mentions: List[MarketMention]) -> Dict[str, Any]:
        """
        Analyze volume trends
        
        Args:
            mentions: Sorted list of mentions by time
            
        Returns:
            Volume trend metrics
        """
        if len(mentions) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate volume over time
        total_volume = len(mentions)
        
        # Split into quarters
        quarter_size = len(mentions) // 4
        quarters = []
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = start_idx + quarter_size if i < 3 else len(mentions)
            quarters.append(mentions[start_idx:end_idx])
        
        quarter_volumes = [len(quarter) for quarter in quarters]
        
        # Calculate trend
        if len(quarter_volumes) >= 2:
            recent_avg = sum(quarter_volumes[-2:]) / 2
            early_avg = sum(quarter_volumes[:2]) / 2
            trend = "increasing" if recent_avg > early_avg else "decreasing" if recent_avg < early_avg else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "total_volume": total_volume,
            "quarter_volumes": quarter_volumes,
            "peak_quarter": quarter_volumes.index(max(quarter_volumes)) if quarter_volumes else None
        }
    
    async def _generate_analysis(self, formatted_mentions: str) -> str:
        """
        Generate AI analysis using the configured provider
        
        Args:
            formatted_mentions: Formatted mentions for analysis
            
        Returns:
            AI-generated analysis response
        """
        prompt = self.trend_prompt_template.format(mentions=formatted_mentions)
        
        if self.model_provider == "openai":
            return await self._analyze_with_openai(prompt)
        elif self.model_provider == "anthropic":
            return await self._analyze_with_anthropic(prompt)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    async def _analyze_with_openai(self, prompt: str) -> str:
        """
        Analyze using OpenAI API
        
        Args:
            prompt: Analysis prompt
            
        Returns:
            AI response
        """
        try:
            import openai
            
            client = openai.AsyncOpenAI(api_key=settings.api.openai_api_key)
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert market trend analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _analyze_with_anthropic(self, prompt: str) -> str:
        """
        Analyze using Anthropic API
        
        Args:
            prompt: Analysis prompt
            
        Returns:
            AI response
        """
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(api_key=settings.api.anthropic_api_key)
            
            response = await client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
