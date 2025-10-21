"""
Main market analysis agent

This module implements the main AI agent that orchestrates multiple
analysis types and provides comprehensive market insights.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .base_agent import BaseAgent, AnalysisResult
from .sentiment_analyzer import SentimentAnalyzer
from .trend_analyzer import TrendAnalyzer
from ..scanner.base_scanner import MarketMention
from ..config import settings


class MarketAgent(BaseAgent):
    """
    Main AI agent for comprehensive market analysis
    
    This agent orchestrates multiple analysis types including sentiment
    analysis, trend analysis, and provides unified market insights.
    """
    
    def __init__(self):
        """Initialize the market agent"""
        super().__init__("Market Agent")
        
        # Initialize sub-agents
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        
        self.comprehensive_prompt_template = """
        Provide a comprehensive market analysis based on the following data:
        
        Market Mentions:
        {mentions}
        
        Sentiment Analysis:
        {sentiment_analysis}
        
        Trend Analysis:
        {trend_analysis}
        
        Please provide:
        1. Executive summary of market conditions
        2. Key market insights and patterns
        3. Risk assessment and opportunities
        4. Market sentiment overview
        5. Trend predictions and implications
        6. Actionable recommendations for investors
        7. Market outlook and confidence levels
        
        Format your response in a structured manner with clear insights and recommendations.
        """
    
    async def analyze(self, mentions: List[MarketMention]) -> AnalysisResult:
        """
        Perform comprehensive market analysis
        
        Args:
            mentions: List of market mentions to analyze
            
        Returns:
            Comprehensive analysis result
        """
        if not mentions:
            return AnalysisResult(
                analysis_type="comprehensive",
                timestamp=datetime.now(),
                data={"error": "No mentions provided"},
                confidence_score=0.0,
                insights=["No data available for market analysis"],
                recommendations=["Collect more market mentions for analysis"]
            )
        
        logger.info(f"Starting comprehensive market analysis for {len(mentions)} mentions")
        
        try:
            # Run sentiment and trend analysis in parallel
            sentiment_task = self.sentiment_analyzer.analyze(mentions)
            trend_task = self.trend_analyzer.analyze(mentions)
            
            sentiment_result, trend_result = await asyncio.gather(sentiment_task, trend_task)
            
            # Generate comprehensive analysis
            comprehensive_analysis = await self._generate_comprehensive_analysis(
                mentions, sentiment_result, trend_result
            )
            
            # Prepare final analysis data
            analysis_data = self._prepare_comprehensive_data(mentions, sentiment_result, trend_result)
            
            # Extract insights and recommendations
            insights = self._extract_insights_from_response(comprehensive_analysis)
            recommendations = self._extract_recommendations_from_response(comprehensive_analysis)
            
            # Calculate overall confidence score
            confidence_score = self._calculate_overall_confidence(sentiment_result, trend_result, len(comprehensive_analysis))
            
            # Combine analysis data
            analysis_data.update({
                "comprehensive_analysis": comprehensive_analysis,
                "sentiment_result": sentiment_result.to_dict(),
                "trend_result": trend_result.to_dict(),
                "total_analyzed": len(mentions)
            })
            
            result = AnalysisResult(
                analysis_type="comprehensive",
                timestamp=datetime.now(),
                data=analysis_data,
                confidence_score=confidence_score,
                insights=insights,
                recommendations=recommendations
            )
            
            logger.info(f"Comprehensive market analysis completed with confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive market analysis: {e}")
            return AnalysisResult(
                analysis_type="comprehensive",
                timestamp=datetime.now(),
                data={"error": str(e)},
                confidence_score=0.0,
                insights=["Market analysis failed"],
                recommendations=["Check AI provider configuration and data quality"]
            )
    
    async def _generate_comprehensive_analysis(self, mentions: List[MarketMention], 
                                             sentiment_result: AnalysisResult, 
                                             trend_result: AnalysisResult) -> str:
        """
        Generate comprehensive AI analysis
        
        Args:
            mentions: List of market mentions
            sentiment_result: Results from sentiment analysis
            trend_result: Results from trend analysis
            
        Returns:
            Comprehensive analysis response
        """
        formatted_mentions = self._format_mentions_for_prompt(mentions)
        
        # Format sub-analysis results
        sentiment_summary = self._format_analysis_summary(sentiment_result)
        trend_summary = self._format_analysis_summary(trend_result)
        
        prompt = self.comprehensive_prompt_template.format(
            mentions=formatted_mentions,
            sentiment_analysis=sentiment_summary,
            trend_analysis=trend_summary
        )
        
        if self.model_provider == "openai":
            return await self._analyze_with_openai(prompt)
        elif self.model_provider == "anthropic":
            return await self._analyze_with_anthropic(prompt)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    def _format_analysis_summary(self, analysis_result: AnalysisResult) -> str:
        """
        Format analysis result into a summary string
        
        Args:
            analysis_result: Analysis result to format
            
        Returns:
            Formatted summary string
        """
        summary = f"Analysis Type: {analysis_result.analysis_type}\n"
        summary += f"Confidence Score: {analysis_result.confidence_score:.2f}\n"
        summary += f"Key Insights:\n"
        
        for i, insight in enumerate(analysis_result.insights[:5], 1):
            summary += f"  {i}. {insight}\n"
        
        summary += f"Recommendations:\n"
        for i, rec in enumerate(analysis_result.recommendations[:3], 1):
            summary += f"  {i}. {rec}\n"
        
        return summary
    
    def _prepare_comprehensive_data(self, mentions: List[MarketMention], 
                                   sentiment_result: AnalysisResult, 
                                   trend_result: AnalysisResult) -> Dict[str, Any]:
        """
        Prepare comprehensive analysis data
        
        Args:
            mentions: List of market mentions
            sentiment_result: Results from sentiment analysis
            trend_result: Results from trend analysis
            
        Returns:
            Comprehensive analysis data
        """
        base_data = self._prepare_data_for_analysis(mentions)
        
        # Add sentiment metrics
        sentiment_data = sentiment_result.data
        if "sentiment_scores" in sentiment_data:
            base_data["sentiment"] = sentiment_data["sentiment_scores"]
        
        # Add trend metrics
        trend_data = trend_result.data
        if "time_trends" in trend_data:
            base_data["trends"] = trend_data
        
        # Calculate overall market health score
        market_health = self._calculate_market_health_score(sentiment_result, trend_result)
        base_data["market_health"] = market_health
        
        return base_data
    
    def _calculate_market_health_score(self, sentiment_result: AnalysisResult, 
                                     trend_result: AnalysisResult) -> Dict[str, Any]:
        """
        Calculate overall market health score
        
        Args:
            sentiment_result: Results from sentiment analysis
            trend_result: Results from trend analysis
            
        Returns:
            Market health score and metrics
        """
        # Extract sentiment score
        sentiment_score = 0.5  # Default neutral
        if "sentiment_scores" in sentiment_result.data:
            sentiment_data = sentiment_result.data["sentiment_scores"]
            if sentiment_data.get("overall") == "very_positive":
                sentiment_score = 0.9
            elif sentiment_data.get("overall") == "positive":
                sentiment_score = 0.7
            elif sentiment_data.get("overall") == "negative":
                sentiment_score = 0.3
            elif sentiment_data.get("overall") == "very_negative":
                sentiment_score = 0.1
        
        # Extract trend score
        trend_score = 0.5  # Default neutral
        if "time_trends" in trend_result.data:
            trend_data = trend_result.data["time_trends"]
            if trend_data.get("trend") == "increasing":
                trend_score = 0.7
            elif trend_data.get("trend") == "decreasing":
                trend_score = 0.3
        
        # Calculate overall health score
        overall_score = (sentiment_score * 0.6 + trend_score * 0.4)
        
        # Determine health level
        if overall_score >= 0.8:
            health_level = "excellent"
        elif overall_score >= 0.6:
            health_level = "good"
        elif overall_score >= 0.4:
            health_level = "moderate"
        elif overall_score >= 0.2:
            health_level = "poor"
        else:
            health_level = "critical"
        
        return {
            "overall_score": overall_score,
            "health_level": health_level,
            "sentiment_score": sentiment_score,
            "trend_score": trend_score,
            "confidence": min(sentiment_result.confidence_score or 0, trend_result.confidence_score or 0)
        }
    
    def _calculate_overall_confidence(self, sentiment_result: AnalysisResult, 
                                    trend_result: AnalysisResult, 
                                    response_length: int) -> float:
        """
        Calculate overall confidence score
        
        Args:
            sentiment_result: Results from sentiment analysis
            trend_result: Results from trend analysis
            response_length: Length of comprehensive response
            
        Returns:
            Overall confidence score
        """
        # Average confidence from sub-analyses
        avg_confidence = (sentiment_result.confidence_score or 0 + trend_result.confidence_score or 0) / 2
        
        # Bonus for detailed response
        response_bonus = min(response_length / 2000, 0.2)  # Up to 20% bonus
        
        return min(avg_confidence + response_bonus, 1.0)
    
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
                    {"role": "system", "content": "You are an expert market analyst with deep knowledge of financial markets, sentiment analysis, and trend identification."},
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
