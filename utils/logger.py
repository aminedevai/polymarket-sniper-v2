"""
utils/logger.py
===============
Structured JSON logging setup.
Outputs one JSON object per line — easy to grep, pipe to jq, or ingest to ELK.
Also keeps a human-readable console handler.

Usage:
    from utils.logger import setup_logging
    setup_logging(level="INFO", log_file="bot.log")
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj)


class HumanFormatter(logging.Formatter):
    COLORS = {
        "DEBUG":    "\033[37m",   # grey
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[1;31m", # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        msg = record.getMessage()
        base = f"{ts} {color}[{record.levelname[0]}]{self.RESET} {record.name}: {msg}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """
    Call once at startup.
    - Console: human-readable with color
    - File (optional): JSON lines for structured analysis
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(HumanFormatter())
    root.addHandler(ch)

    # File handler (JSON)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    for noisy in ("websockets", "asyncio", "aiohttp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("main").info(
        f"Logging initialized | level={level} | file={log_file or 'none'}"
    )
