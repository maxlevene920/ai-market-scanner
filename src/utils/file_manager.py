"""
File management utilities for the AI Market Scanner

This module provides utilities for managing files, directories,
and data persistence for the application.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
from loguru import logger

from ..scanner.base_scanner import MarketMention
from ..agent.base_agent import AnalysisResult


class FileManager:
    """
    Utility class for file and directory management
    
    This class provides methods for managing application files,
    creating directories, and handling data persistence.
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize the file manager
        
        Args:
            base_dir: Base directory for application data
        """
        self.base_dir = Path(base_dir)
        self.logger = logger.bind(name="FileManager")
        
        # Create base directory structure
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create the application directory structure"""
        directories = [
            self.base_dir,
            self.base_dir / "mentions",
            self.base_dir / "analysis",
            self.base_dir / "exports",
            self.base_dir / "cache",
            self.base_dir / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created directory structure in {self.base_dir}")
    
    def save_mentions(self, mentions: List[MarketMention], filename: Optional[str] = None) -> str:
        """
        Save market mentions to file
        
        Args:
            mentions: List of market mentions to save
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mentions_{timestamp}.json"
        
        file_path = self.base_dir / "mentions" / filename
        
        try:
            # Convert mentions to dictionaries
            data = [mention.to_dict() for mention in mentions]
            
            # Save to JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Saved {len(mentions)} mentions to {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving mentions: {e}")
            raise
    
    def load_mentions(self, filename: str) -> List[MarketMention]:
        """
        Load market mentions from file
        
        Args:
            filename: Name of the file to load
            
        Returns:
            List of market mentions
        """
        file_path = self.base_dir / "mentions" / filename
        
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
            self.logger.error(f"Error loading mentions: {e}")
            return []
    
    def save_analysis(self, analysis_result: AnalysisResult, filename: Optional[str] = None) -> str:
        """
        Save analysis result to file
        
        Args:
            analysis_result: Analysis result to save
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.json"
        
        file_path = self.base_dir / "analysis" / filename
        
        try:
            # Convert to dictionary and save
            data = analysis_result.to_dict()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Saved analysis result to {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving analysis: {e}")
            raise
    
    def load_analysis(self, filename: str) -> Optional[AnalysisResult]:
        """
        Load analysis result from file
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Analysis result or None if failed
        """
        file_path = self.base_dir / "analysis" / filename
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct AnalysisResult from dictionary
            analysis_result = AnalysisResult(
                analysis_type=data["analysis_type"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                data=data["data"],
                confidence_score=data.get("confidence_score"),
                insights=data.get("insights", []),
                recommendations=data.get("recommendations", [])
            )
            
            self.logger.info(f"Loaded analysis result from {file_path}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error loading analysis: {e}")
            return None
    
    def list_mentions_files(self) -> List[str]:
        """
        List all mentions files
        
        Returns:
            List of mentions filenames
        """
        mentions_dir = self.base_dir / "mentions"
        if not mentions_dir.exists():
            return []
        
        files = [f.name for f in mentions_dir.glob("*.json")]
        return sorted(files, reverse=True)  # Most recent first
    
    def list_analysis_files(self) -> List[str]:
        """
        List all analysis files
        
        Returns:
            List of analysis filenames
        """
        analysis_dir = self.base_dir / "analysis"
        if not analysis_dir.exists():
            return []
        
        files = [f.name for f in analysis_dir.glob("*.json")]
        return sorted(files, reverse=True)  # Most recent first
    
    def delete_file(self, file_type: str, filename: str) -> bool:
        """
        Delete a file
        
        Args:
            file_type: Type of file ("mentions" or "analysis")
            filename: Name of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if file_type == "mentions":
                file_path = self.base_dir / "mentions" / filename
            elif file_type == "analysis":
                file_path = self.base_dir / "analysis" / filename
            else:
                self.logger.error(f"Invalid file type: {file_type}")
                return False
            
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"Deleted file: {file_path}")
                return True
            else:
                self.logger.warning(f"File not found: {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting file: {e}")
            return False
    
    def cleanup_old_files(self, days: int = 30) -> int:
        """
        Clean up old files
        
        Args:
            days: Number of days to keep files
            
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        # Clean up mentions files
        mentions_dir = self.base_dir / "mentions"
        if mentions_dir.exists():
            for file_path in mentions_dir.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        self.logger.info(f"Deleted old file: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error deleting file {file_path}: {e}")
        
        # Clean up analysis files
        analysis_dir = self.base_dir / "analysis"
        if analysis_dir.exists():
            for file_path in analysis_dir.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        self.logger.info(f"Deleted old file: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error deleting file {file_path}: {e}")
        
        self.logger.info(f"Cleaned up {deleted_count} old files")
        return deleted_count
    
    def get_file_info(self, file_type: str, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a file
        
        Args:
            file_type: Type of file ("mentions" or "analysis")
            filename: Name of the file
            
        Returns:
            Dictionary with file information or None if not found
        """
        try:
            if file_type == "mentions":
                file_path = self.base_dir / "mentions" / filename
            elif file_type == "analysis":
                file_path = self.base_dir / "analysis" / filename
            else:
                return None
            
            if not file_path.exists():
                return None
            
            stat = file_path.stat()
            
            return {
                "filename": filename,
                "file_type": file_type,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "path": str(file_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting file info: {e}")
            return None
    
    def export_to_csv(self, mentions: List[MarketMention], filename: Optional[str] = None) -> str:
        """
        Export mentions to CSV file
        
        Args:
            mentions: List of market mentions to export
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to exported CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mentions_{timestamp}.csv"
        
        file_path = self.base_dir / "exports" / filename
        
        try:
            import pandas as pd
            
            # Convert mentions to dictionaries
            data = [mention.to_dict() for mention in mentions]
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            self.logger.info(f"Exported {len(mentions)} mentions to CSV: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            raise
    
    def get_directory_size(self) -> Dict[str, Any]:
        """
        Get size information for the base directory
        
        Returns:
            Dictionary with directory size information
        """
        total_size = 0
        file_count = 0
        directory_sizes = {}
        
        for subdir in ["mentions", "analysis", "exports", "cache", "logs"]:
            dir_path = self.base_dir / subdir
            if dir_path.exists():
                dir_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                dir_files = len(list(dir_path.rglob('*')))
                directory_sizes[subdir] = {
                    "size_bytes": dir_size,
                    "size_mb": round(dir_size / (1024 * 1024), 2),
                    "file_count": dir_files
                }
                total_size += dir_size
                file_count += dir_files
        
        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_files": file_count,
            "directories": directory_sizes
        }
