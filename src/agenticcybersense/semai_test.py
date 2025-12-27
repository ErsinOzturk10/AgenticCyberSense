"""Dummy python file for semai_test module."""

import logging

logger = logging.getLogger(__name__)


def dummy_function() -> None:
    """Log a dummy message."""
    logger.debug("This is a dummy function.")
