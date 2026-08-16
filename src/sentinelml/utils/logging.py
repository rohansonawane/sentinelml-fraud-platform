from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

import structlog


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def bind_request_id(request_id: str | None = None) -> str:
    rid = request_id or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=rid)
    return rid


def get_logger(name: str, **kwargs: Any) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name).bind(**kwargs)
