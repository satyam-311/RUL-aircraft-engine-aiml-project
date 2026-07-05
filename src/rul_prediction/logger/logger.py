"""
Centralized logging setup for the RUL prediction project.

WHY: print() statements don't persist, can't be filtered by severity, and
give no timestamp/context. In production ML systems, every pipeline stage
(data loading, training, inference) needs to write to a persistent,
timestamped log so failures can be debugged after the fact.

HOW: Uses Python's built-in `logging` module with a RotatingFileHandler
(caps log file size so logs don't grow unbounded) plus a console handler
for real-time feedback during development.

WHERE: Imported by every other module in the project via
    from rul_prediction.logger.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[3] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "rul_prediction.log"

LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger instance.

    Args:
        name: Usually __name__ of the calling module, so logs show origin.
        level: Minimum severity level to log (default INFO).

    Returns:
        A logging.Logger with both file and console handlers attached.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Avoid duplicate handlers if get_logger() is called multiple times
        return logger

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
