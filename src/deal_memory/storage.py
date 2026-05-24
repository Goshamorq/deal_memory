"""JSONL read/write — shared by CLI, eval and UI."""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> int:
    """Write pydantic models as JSONL. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
            count += 1
    return count


def read_jsonl(path: Path, model: type[T]) -> Iterator[T]:
    """Stream JSONL as pydantic models."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield model.model_validate_json(line)


def load_jsonl(path: Path, model: type[T]) -> list[T]:
    """Load whole JSONL into a list. Convenience for UI/eval."""
    return list(read_jsonl(path, model))


def list_eval_runs(eval_dir: Path) -> list[Path]:
    """Return all eval-run JSONL files, newest first."""
    if not eval_dir.exists():
        return []
    return sorted(eval_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
