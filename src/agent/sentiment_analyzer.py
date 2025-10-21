"""
Sentiment analysis agent for market data

This module implements an AI agent that analyzes sentiment
in market mentions and provides sentiment-based insights.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .base_agent import BaseAgent, AnalysisResult
from ..scanner.base_scanner import MarketMention
from ..config import settings


class SentimentAnalyzer(BaseAgent):
    """
    AI agent for sentiment analysis of market mentions
    
    This agent analyzes the sentiment of market mentions and provides
    insights about market mood, investor sentiment, and emotional trends.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        super().__init__("Sentiment Analyzer")
        self.sentiment_prompt_template = """
        Analyze the sentiment of the following market mentions and provide insights:
        
        Market Mentions:
        {mentions}
        
        Please provide:
        1. Overall market sentiment (positive, negative, neutral)
        2. Sentiment breakdown by source
        3. Key emotional indicators
        4. Sentiment trends over time
        5. Potential market implications
        
        Format your response in a structured manner with clear insights and recommendations.
        """
    
    async def analyze(self, mentions: List[MarketMention]) -> AnalysisResult:
        """
        Analyze sentiment of market mentions
        
        Args:
            mentions: List of market mentions to analyze
            
        Returns:
            Analysis result containing sentiment insights
        """
        if not mentions:
            return AnalysisResult(
                analysis_type="sentiment",
                timestamp=datetime.now(),
                data={"error": "No mentions provided"},
                confidence_score=0.0,
                insights=["No data available for sentiment analysis"],
                recommendations=["Collect more market mentions for analysis"]
            )
        
        logger.info(f"Analyzing sentiment for {len(mentions)} mentions")
        
        # Prepare data for analysis
        analysis_data = self._prepare_data_for_analysis(mentions)
        formatted_mentions = self._format_mentions_for_prompt(mentions)
        
        # Generate AI analysis
        try:
            response = await self._generate_analysis(formatted_mentions)
            
            # Extract insights and recommendations
            insights = self._extract_insights_from_response(response)
            recommendations = self._extract_recommendations_from_response(response)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(mentions, len(response))
            
            # Calculate sentiment scores
            sentiment_scores = self._calculate_sentiment_scores(mentions)
            
            # Combine analysis data
            analysis_data.update({
                "sentiment_scores": sentiment_scores,
                "ai_response": response,
                "total_analyzed": len(mentions)
            })
            
            result = AnalysisResult(
                analysis_type="sentiment",
                timestamp=datetime.now(),
                data=analysis_data,
                confidence_score=confidence_score,
                insights=insights,
                recommendations=recommendations
            )
            
            logger.info(f"Sentiment analysis completed with confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return AnalysisResult(
                analysis_type="sentiment",
                timestamp=datetime.now(),
                data={"error": str(e)},
                confidence_score=0.0,
                insights=["Sentiment analysis failed"],
                recommendations=["Check AI provider configuration"]
            )
    
    async def _generate_analysis(self, formatted_mentions: str) -> str:
        """
        Generate AI analysis using the configured provider
        
        Args:
            formatted_mentions: Formatted mentions for analysis
            
        Returns:
            AI-generated analysis response
        """
        prompt = self.sentiment_prompt_template.format(mentions=formatted_mentions)
        
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
                    {"role": "system", "content": "You are an expert market sentiment analyst."},
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
    
    def _calculate_sentiment_scores(self, mentions: List[MarketMention]) -> Dict[str, Any]:
        """
        Calculate sentiment scores from mentions
        
        Args:
            mentions: List of market mentions
            
        Returns:
            Dictionary with sentiment scores
        """
        # Simple keyword-based sentiment analysis
        positive_keywords = [
            "bullish", "growth", "profit", "gain", "rise", "increase", "positive",
            "optimistic", "strong", "good", "excellent", "outperform", "buy"
        ]
        
        negative_keywords = [
            "bearish", "decline", "loss", "fall", "drop", "decrease", "negative",
            "pessimistic", "weak", "bad", "poor", "underperform", "sell"
        ]
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for mention in mentions:
            content_lower = mention.content.lower()
            
            pos_matches = sum(1 for keyword in positive_keywords if keyword in content_lower)
            neg_matches = sum(1 for keyword in negative_keywords if keyword in content_lower)
            
            if pos_matches > neg_matches:
                positive_count += 1
            elif neg_matches > pos_matches:
                negative_count += 1
            else:
                neutral_count += 1
        
        total = len(mentions)
        if total == 0:
            return {"positive": 0, "negative": 0, "neutral": 0, "overall": "neutral"}
        
        return {
            "positive": positive_count / total,
            "negative": negative_count / total,
            "neutral": neutral_count / total,
            "overall": self._determine_overall_sentiment(positive_count, negative_count, neutral_count)
        }
    
    def _determine_overall_sentiment(self, positive: int, negative: int, neutral: int) -> str:
        """
        Determine overall sentiment from counts
        
        Args:
            positive: Number of positive mentions
            negative: Number of negative mentions
            neutral: Number of neutral mentions
            
        Returns:
            Overall sentiment label
        """
        total = positive + negative + neutral
        if total == 0:
            return "neutral"
        
        pos_ratio = positive / total
        neg_ratio = negative / total
        
        if pos_ratio > 0.6:
            return "very_positive"
        elif pos_ratio > 0.4:
            return "positive"
        elif neg_ratio > 0.6:
            return "very_negative"
        elif neg_ratio > 0.4:
            return "negative"
        else:
            return "neutral"
