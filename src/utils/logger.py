"""
Logging configuration and utilities

This module provides centralized logging configuration for the
AI Market Scanner application with structured logging support.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Remove default logger
    logger.remove()
    
    # Console logging format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler if specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File logging format
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        logger.add(
            log_file,
            format=file_format,
            level=log_level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    # Add error file handler
    error_log_file = "logs/errors.log"
    Path(error_log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        error_log_file,
        format=file_format,
        level="ERROR",
        rotation="5 MB",
        retention="60 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    logger.info(f"Logging configured with level: {log_level}")


def get_logger(name: str = None):
    """
    Get a logger instance
    
    Args:
        name: Optional logger name
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class
    
    This mixin provides easy access to logging functionality
    for any class that inherits from it.
    """
    
    @property
    def logger(self):
        """Get logger instance for this class"""
        return get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def log_exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, **kwargs)


def log_function_call(func):
    """
    Decorator to log function calls
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    return wrapper


def log_async_function_call(func):
    """
    Decorator to log async function calls
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated async function
    """
    async def wrapper(*args, **kwargs):
        logger.debug(f"Calling async {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Async {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Async {func.__name__} failed with error: {e}")
            raise
    return wrapper


class PerformanceLogger:
    """
    Context manager for logging performance metrics
    
    This class provides a context manager to measure and log
    the execution time of code blocks.
    """
    
    def __init__(self, operation_name: str, logger_instance=None):
        """
        Initialize performance logger
        
        Args:
            operation_name: Name of the operation being measured
            logger_instance: Optional logger instance
        """
        self.operation_name = operation_name
        self.logger = logger_instance or get_logger("Performance")
        self.start_time = None
    
    def __enter__(self):
        """Enter context manager"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        if self.start_time:
            duration = datetime.now() - self.start_time
            if exc_type:
                self.logger.error(f"{self.operation_name} failed after {duration.total_seconds():.2f}s")
            else:
                self.logger.info(f"{self.operation_name} completed in {duration.total_seconds():.2f}s")
    
    def log_progress(self, message: str):
        """Log progress message"""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            self.logger.info(f"{self.operation_name} - {message} (elapsed: {elapsed.total_seconds():.2f}s)")


# Initialize logging on module import
setup_logging()
