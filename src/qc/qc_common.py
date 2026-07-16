"""Shared helpers for the Phase-0 data-quality QC scripts (src/qc/).

These scripts are standalone (run as `python src/qc/<name>.py`, astro env) and deliberately do NOT
touch the locked Stage-0d builder `src/build_variability_labels.py`. They only read the canonical
label CSV + catalogs and write to `labels/qc/` (or the versioned `labels/variability_labels_star_v2.csv`).

Kept minimal on purpose: project-root detection, logging, and a network retry wrapper matching the
`/astroquery-resilience` policy (tenacity-style backoff on connection/timeout errors).
"""
from __future__ import annotations

import logging
import socket
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

_BACKOFF = (5, 15, 45)  # waits (s) before attempts 2, 3, 4


def find_project_root() -> Path:
    """Walk up from CWD until CLAUDE.md is found (mirrors build_variability_labels.find_project_root)."""
    p = Path.cwd()
    for _ in range(10):
        if (p / "CLAUDE.md").exists():
            return p
        p = p.parent
    raise FileNotFoundError("CLAUDE.md not found — cannot determine project root")


def setup_logging(log_file: Path, name: str) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def call_with_retry(func: Callable[[], T], label: str, logger: logging.Logger) -> T:
    """Run `func`, retrying transient network failures with fixed backoff. Reraises on the last attempt."""
    try:
        import requests.exceptions as rexc
        import urllib3.exceptions as u3exc
        net_errors: tuple = (
            rexc.ConnectionError, rexc.Timeout, rexc.ChunkedEncodingError,
            u3exc.ProtocolError, socket.gaierror, socket.timeout,
            ConnectionError, TimeoutError,
        )
    except Exception:  # pragma: no cover - requests/urllib3 always present in astro
        net_errors = (socket.gaierror, socket.timeout, ConnectionError, TimeoutError)

    for attempt in range(1, len(_BACKOFF) + 2):
        try:
            return func()
        except net_errors as e:  # type: ignore[misc]
            if attempt > len(_BACKOFF):
                raise
            wait = _BACKOFF[attempt - 1]
            logger.warning(f"{label} attempt {attempt} failed: {type(e).__name__}: {e}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover
