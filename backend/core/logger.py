# backend/core/logging.py
import logging
import sys

from backend.core.config import get_settings


def setup_logging() -> None:
    """
    Configure root logging for the whole app.
    Call this once, at startup, from app.py.
    """
    settings = get_settings()
    level = logging.DEBUG if settings.debug else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging() is accidentally called twice
    # (e.g. with --reload triggering re-imports)
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Quiet down noisy third-party loggers so app logs aren't drowned out
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Usage in any file:
        from backend.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Detector loaded")
    """
    return logging.getLogger(name)