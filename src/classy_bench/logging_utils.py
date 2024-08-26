"""Logging utilities."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(filename)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
)

logger = logging.getLogger()
