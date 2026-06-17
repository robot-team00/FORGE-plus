"""Structured logging for Force-Budgeted Recovery experiments."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {
            "level": record.levelname,
            "time": self.formatTime(record, self.datefmt),
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d)


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: str | None = None,
) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    fmt = JSONFormatter() if json_format else logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for h in handlers:
        h.setFormatter(fmt)

    logging.basicConfig(level=getattr(logging, level.upper()), handlers=handlers)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
