"""
Logging helpers to make stage outputs consistent and readable.
"""
import logging


def stage_info(logger: logging.Logger, stage: str, message: str):
    logger.info(f"[STAGE:{stage}] {message}")


def stage_warn(logger: logging.Logger, stage: str, message: str):
    logger.warning(f"[STAGE:{stage}] {message}")


def stage_error(logger: logging.Logger, stage: str, message: str):
    logger.error(f"[STAGE:{stage}] {message}")
