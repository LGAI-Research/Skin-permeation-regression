"""Skin permeation benchmark construction package."""

from __future__ import annotations

from typing import Any


def run_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Lazily import the merge pipeline entry point."""

    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


def run_benchmark(*args: Any, **kwargs: Any) -> Any:
    """Lazily import the benchmarking entry point."""

    from .benchmarking import run_benchmark as _run_benchmark

    return _run_benchmark(*args, **kwargs)


__all__ = ["run_pipeline", "run_benchmark"]
