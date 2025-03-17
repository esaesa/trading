# logger_config.py

import logging
from colorlog import ColoredFormatter
from logging.handlers import RotatingFileHandler
import os

# 1) Name your logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 2) Console handler (same as before)
console_handler = logging.StreamHandler()
console_formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)
console_handler.setFormatter(console_formatter)

# 3) Create a folder for logs (optional)
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
log_file_path = os.path.join(log_folder, "app.log")

# 4) File handler with rotation
file_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=1_000_000,   # e.g., 1 MB
    backupCount=3,        # keep 3 old log files
    encoding="utf-8"
)
# 5) (Optional) a simpler log format for file logs
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
file_handler.setFormatter(file_formatter)

# 6) Add handlers only if no handlers exist
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
