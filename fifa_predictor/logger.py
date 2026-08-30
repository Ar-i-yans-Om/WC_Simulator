"""
logger.py
=========

Centralised logging configuration for the FIFA WC 2026 prediction engine.

Call setup() once at the start of match_runner.py or main.py.
All agents use:

    import logging
    logger = logging.getLogger(__name__)

and log with logger.info() / logger.debug().

The format shows time + a clean message, e.g.:

    [14:32:01] [Researcher] Mexico — scored 1.6 / conceded 1.1 · fitness ×0.97
    [14:32:03] [Alchemist-A] Mexico — cohesion ×1.06 (2 clusters: Real Madrid×3, PSG×2)
    [14:32:05] [Strategist-A] Mexico → Full Intensity ×1.05 | "MD1 — must win points."
"""

from __future__ import annotations

import logging
import sys


_FMT  = "[%(asctime)s]  %(message)s"
_DATE = "%H:%M:%S"

# Third-party libraries that spam at INFO level
_QUIET = [
    "httpx", "httpcore", "urllib3", "urllib3.connectionpool",
    "google", "google.auth", "google.auth.transport",
    "langgraph", "langchain", "langchain_core",
]


def setup(verbose: bool = False) -> None:
    """
    Configure root logger.  Call once before invoking run_prediction().

    verbose=True  → DEBUG level (very noisy, shows all internal library logs)
    verbose=False → INFO level  (agent decisions only — the default)
    """
    # On Windows, sys.stdout/stderr default to the cp1252 codepage when output is
    # redirected (to a file/pipe). The agent logs and print_result() emit Unicode
    # (→ ✗ ─ ✎ ⚡ …), and a single un-encodable char raises UnicodeEncodeError that
    # aborts the whole fixture AFTER a successful prediction. Force UTF-8 with
    # errors="replace" so output can never crash a run. (Python 3.7+ .reconfigure.)
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format=_FMT,
        datefmt=_DATE,
        stream=sys.stdout,
        force=True,           # override any earlier basicConfig call
    )

    # Keep third-party libraries quiet at INFO level
    for name in _QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)
