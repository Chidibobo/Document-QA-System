"""
Centralized logging configuration
"""
import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime


class LoggerSetup:
    """Setup and configure logging for the application"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def get_logger(self, name, level=logging.INFO):
        """
        Get a configured logger instance
        
        Args:
            name (str): Name of the logger (usually __name__ of the module)
            level: Logging level (default: INFO)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
            
        logger.setLevel(level)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler - INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # File Handler - All logs (rotating file)
        log_file = self.log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Error File Handler - ERROR and above only
        error_log_file = self.log_dir / f"error_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
        
        return logger


# Global logger setup instance
_logger_setup = LoggerSetup()


def get_logger(name, level=logging.INFO):
    """
    Convenience function to get a logger
    
    Usage:
        from config.logger import get_logger
        logger = get_logger(__name__)
        logger.info("This is an info message")
    """
    return _logger_setup.get_logger(name, level)

