"""Stderr-only logging.

The MCP server speaks JSON-RPC over stdout, so nothing in this package may log to stdout.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger("rundoctor")
        root.addHandler(handler)
        root.setLevel(os.environ.get("RUNDOCTOR_LOG_LEVEL", "INFO"))
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("rundoctor") else f"rundoctor.{name}")
