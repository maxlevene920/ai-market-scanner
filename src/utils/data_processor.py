"""
Data processing utilities for market analysis

This module provides utilities for processing, cleaning, and
transforming market data for analysis.
"""

import json
import csv
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import pandas as pd
from loguru import logger

from ..scanner.base_scanner import MarketMention
from ..agent.base_agent import AnalysisResult


class DataProcessor:
    """
    Utility class for processing market data
    
    This class provides methods for cleaning, transforming,
    and exporting market data and analysis results.
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.logger = logger.bind(name="DataProcessor")
    
    def clean_mentions(self, mentions: List[MarketMention]) -> List[MarketMention]:
        """
        Clean and validate market mentions
        
        Args:
            mentions: List of market mentions to clean
            
        Returns:
            Cleaned list of market mentions
        """
        cleaned_mentions = []
        
        for mention in mentions:
            try:
                # Clean text content
                mention.title = self._clean_text(mention.title)
                mention.content = self._clean_text(mention.content)
                
                # Validate required fields
                if self._validate_mention(mention):
                    cleaned_mentions.append(mention)
                else:
                    self.logger.warning(f"Invalid mention skipped: {mention.title[:50]}...")
                    
            except Exception as e:
                self.logger.error(f"Error cleaning mention: {e}")
                continue
        
        self.logger.info(f"Cleaned {len(cleaned_mentions)} out of {len(mentions)} mentions")
        return cleaned_mentions
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text content
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common noise characters
        noise_chars = ["\n", "\r", "\t", "\u00a0"]
        for char in noise_chars:
            text = text.replace(char, " ")
        
        # Remove excessive punctuation
        text = text.replace("...", ".")
        text = text.replace("!!", "!")
        text = text.replace("??", "?")
        
        return text.strip()
    
    def _validate_mention(self, mention: MarketMention) -> bool:
        """
        Validate a market mention
        
        Args:
            mention: Market mention to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not mention.title or len(mention.title.strip()) < 5:
            return False
        
        if not mention.content or len(mention.content.strip()) < 10:
            return False
        
        if not mention.url or not mention.url.startswith(("http://", "https://")):
            return False
        
        if not mention.source or len(mention.source.strip()) < 2:
            return False
        
        return True
    
    def export_mentions_to_csv(self, mentions: List[MarketMention], file_path: str) -> bool:
        """
        Export market mentions to CSV file
        
        Args:
            mentions: List of market mentions to export
            file_path: Path to save CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert mentions to dictionaries
            data = [mention.to_dict() for mention in mentions]
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            self.logger.info(f"Exported {len(mentions)} mentions to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting mentions to CSV: {e}")
            return False
    
    def export_mentions_to_json(self, mentions: List[MarketMention], file_path: str) -> bool:
        """
        Export market mentions to JSON file
        
        Args:
            mentions: List of market mentions to export
            file_path: Path to save JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert mentions to dictionaries
            data = [mention.to_dict() for mention in mentions]
            
            # Save to JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Exported {len(mentions)} mentions to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting mentions to JSON: {e}")
            return False
    
    def export_analysis_to_json(self, analysis_result: AnalysisResult, file_path: str) -> bool:
        """
        Export analysis result to JSON file
        
        Args:
            analysis_result: Analysis result to export
            file_path: Path to save JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary and save
            data = analysis_result.to_dict()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Exported analysis result to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis to JSON: {e}")
            return False
    
    def load_mentions_from_json(self, file_path: str) -> List[MarketMention]:
        """
        Load market mentions from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of market mentions
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            mentions = []
            for item in data:
                try:
                    mention = MarketMention.from_dict(item)
                    mentions.append(mention)
                except Exception as e:
                    self.logger.warning(f"Error loading mention: {e}")
                    continue
            
            self.logger.info(f"Loaded {len(mentions)} mentions from {file_path}")
            return mentions
            
        except Exception as e:
            self.logger.error(f"Error loading mentions from JSON: {e}")
            return []
    
    def filter_mentions_by_date(self, mentions: List[MarketMention], 
                               start_date: datetime, end_date: datetime) -> List[MarketMention]:
        """
        Filter mentions by date range
        
        Args:
            mentions: List of market mentions
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Filtered list of mentions
        """
        filtered_mentions = []
        
        for mention in mentions:
            if start_date <= mention.published_at <= end_date:
                filtered_mentions.append(mention)
        
        self.logger.info(f"Filtered {len(filtered_mentions)} mentions by date range")
        return filtered_mentions
    
    def filter_mentions_by_relevance(self, mentions: List[MarketMention], 
                                   min_relevance: float) -> List[MarketMention]:
        """
        Filter mentions by minimum relevance score
        
        Args:
            mentions: List of market mentions
            min_relevance: Minimum relevance score threshold
            
        Returns:
            Filtered list of mentions
        """
        filtered_mentions = []
        
        for mention in mentions:
            if mention.relevance_score and mention.relevance_score >= min_relevance:
                filtered_mentions.append(mention)
        
        self.logger.info(f"Filtered {len(filtered_mentions)} mentions by relevance >= {min_relevance}")
        return filtered_mentions
    
    def group_mentions_by_source(self, mentions: List[MarketMention]) -> Dict[str, List[MarketMention]]:
        """
        Group mentions by source
        
        Args:
            mentions: List of market mentions
            
        Returns:
            Dictionary mapping sources to lists of mentions
        """
        grouped = {}
        
        for mention in mentions:
            source = mention.source
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(mention)
        
        self.logger.info(f"Grouped mentions into {len(grouped)} sources")
        return grouped
    
    def calculate_mention_statistics(self, mentions: List[MarketMention]) -> Dict[str, Any]:
        """
        Calculate statistics for market mentions
        
        Args:
            mentions: List of market mentions
            
        Returns:
            Dictionary with calculated statistics
        """
        if not mentions:
            return {"total": 0}
        
        # Basic counts
        total_mentions = len(mentions)
        
        # Relevance statistics
        relevance_scores = [m.relevance_score for m in mentions if m.relevance_score is not None]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Source distribution
        source_counts = {}
        for mention in mentions:
            source_counts[mention.source] = source_counts.get(mention.source, 0) + 1
        
        # Keyword frequency
        keyword_counts = {}
        for mention in mentions:
            for keyword in mention.keywords_found:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Time range
        dates = [m.published_at for m in mentions]
        earliest_date = min(dates) if dates else None
        latest_date = max(dates) if dates else None
        
        statistics = {
            "total_mentions": total_mentions,
            "average_relevance": avg_relevance,
            "source_distribution": source_counts,
            "top_keywords": sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "time_range": {
                "earliest": earliest_date.isoformat() if earliest_date else None,
                "latest": latest_date.isoformat() if latest_date else None
            }
        }
        
        self.logger.info(f"Calculated statistics for {total_mentions} mentions")
        return statistics
