# cinfer/core/logging.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import get_config_manager

def setup_logging():
    """
    Configures logging for the application based on settings from ConfigManager.
    """
    config = get_config_manager()
    log_level_str = config.get_config("server.log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    log_format = config.get_config("logging.format", "%(levelname)s - %(asctime)s - %(name)s - %(module)s:%(lineno)d - %(message)s")
    date_format = config.get_config("logging.date_format", "%Y-%m-%d %H:%M:%S")

    # Create a root logger
    root_logger = logging.getLogger("cinfer") # Base logger for the app
    root_logger.setLevel(log_level)
    root_logger.handlers = [] # Clear any existing handlers (e.g., from basicConfig)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File Handler (Optional, based on config)
    log_file_path_str = config.get_config("logging.file.path", "data/logs/log")
    if log_file_path_str:
        log_file_path = Path(log_file_path_str)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        max_bytes = config.get_config("logging.file.max_bytes", 1024 * 1024 * 5) # 5MB
        backup_count = config.get_config("logging.file.backup_count", 5)

        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"File logging configured at: {log_file_path}")

    # Set level for other common loggers if needed
    logging.getLogger("uvicorn.error").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING if log_level > logging.INFO else log_level)


    root_logger.info(f"Logging setup complete. Application log level set to {log_level_str}.")
    # Example to test:
    # logger = logging.getLogger("main")
    # logger.debug("This is a debug message.")
    # logger.info("This is an info message.")
    # logger.warning("This is a warning message.")
    # logger.error("This is an error message.")

# You would call setup_logging() once when your application starts.
# In your config/config.yaml, you might add a logging section:
# logging:
#   format: "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
#   date_format: "%Y-%m-%d %H:%M:%S"
#   file:
#     path: "data/logs/log" # Set to null or remove to disable file logging
#     max_bytes: 5242880 # 5MB
#     backup_count: 5