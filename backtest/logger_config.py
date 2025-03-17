# logger_config.py

import logging
from colorlog import ColoredFormatter
from logging.handlers import RotatingFileHandler
import os
from rich.console import Console
from rich.table import Table

class RichColoredFormatter(ColoredFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a reusable Console instance with recording enabled.
        self.console = Console(record=True)

    def format(self, record):
        # Check if record.msg is a Rich renderable (e.g., Table) by checking export_text support.
        if isinstance(record.msg, Table) or hasattr(record.msg, "export_text"):
            # Capture the rendered output instead of printing immediately.
            self.console.begin_capture()
            self.console.print(record.msg)
            captured_text = self.console.end_capture()
            record.msg = "\n" + captured_text
        return super().format(record)

# 1) Name your logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Disable propagation to avoid duplicates

# 2) Console handler using our custom formatter
console_handler = logging.StreamHandler()
console_formatter = RichColoredFormatter(
    "%(log_color)s%(blue)s%(message)s",
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
log_folder = os.path.dirname(__file__) 
os.makedirs(log_folder, exist_ok=True)
log_file_path = os.path.join(log_folder, "logs/app.log")

# 4) File handler with rotation
file_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=1_000_000,   # e.g., 1 MB
    backupCount=3,        # keep 3 old log files
    encoding="utf-8"
)
# 5) File formatter (unchanged)
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
file_handler.setFormatter(file_formatter)

# 6) Add handlers only if no handlers exist
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
