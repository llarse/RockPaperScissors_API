################################################################
# USAGE:
# from core.logging_config import app_logger
################################################################


import logging
import sys
from logging.handlers import RotatingFileHandler

# Define the basic configuration for logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "{asctime} - {levelname} - {name} - {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "default",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3,
            "encoding": "utf8",
        },
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
        },
        "fastapi": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
        "app": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
        },
    },
}

# Apply the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Create a custom logger for the application
app_logger = logging.getLogger("app")
