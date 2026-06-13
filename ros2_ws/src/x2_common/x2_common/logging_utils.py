"""Structured logging helpers. Logging is part of the feature, not an afterthought.

Provides a plain-Python logger (works in tests / offline tools) plus a JSON-lines record
writer for the per-test logs required by the test protocols (roadmap §9.4). No ROS2
dependency; nodes may also forward through ``node.get_logger()`` as usual.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, TextIO

_DEF_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, attaching a stream handler once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEF_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class JsonlRecorder:
    """Append structured records as JSON lines to a file.

    Use for saved test logs so that runs are machine-analysable
    (``tools/analyze_policy_log.py`` / ``compare_sim_real.py``). One JSON object per line.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any], add_wall_time: bool = True) -> None:
        if add_wall_time and "wall_time" not in record:
            record = {"wall_time": time.time(), **record}
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    """Best-effort serialisation for numpy arrays / scalars in log records."""
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(obj, "item", None)
    if callable(item):
        return item()
    return str(obj)
