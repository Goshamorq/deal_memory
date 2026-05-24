"""Annotation storage: per-pool JSONL of user verdicts on extractor predictions.

One file per pool: data/annotations/<pool>.jsonl.
One row per (dialog_id, field) — newest write wins on conflict.
"""
from __future__ import annotations

from pathlib import Path

from deal_memory.schema import FIELD_KEYS, Annotation, Verdict
from deal_memory.storage import load_jsonl, write_jsonl


def annotations_path(pool: str, base: Path = Path("data/annotations")) -> Path:
    return base / f"{pool}.jsonl"


def load_pool(pool: str, base: Path = Path("data/annotations")) -> dict[tuple[str, str], Annotation]:
    """Return {(dialog_id, field): Annotation} — latest verdict per cell."""
    path = annotations_path(pool, base)
    if not path.exists():
        return {}
    out: dict[tuple[str, str], Annotation] = {}
    for ann in load_jsonl(path, Annotation):
        out[(ann.dialog_id, ann.field)] = ann  # later rows overwrite earlier
    return out


def upsert(
    pool: str,
    dialog_id: str,
    field: str,
    verdict: Verdict,
    base: Path = Path("data/annotations"),
) -> Annotation:
    """Append a new verdict; latest wins on next load."""
    if field not in FIELD_KEYS:
        raise ValueError(f"Unknown field: {field!r}. Allowed: {FIELD_KEYS}")
    current = load_pool(pool, base)
    ann = Annotation(dialog_id=dialog_id, field=field, verdict=verdict)
    current[(dialog_id, field)] = ann
    # Rewrite the whole file in deterministic order — small enough not to care.
    write_jsonl(annotations_path(pool, base), current.values())
    return ann


def clear(
    pool: str,
    dialog_id: str,
    field: str,
    base: Path = Path("data/annotations"),
) -> None:
    """Remove a verdict for (dialog_id, field) if it exists."""
    current = load_pool(pool, base)
    current.pop((dialog_id, field), None)
    if current:
        write_jsonl(annotations_path(pool, base), current.values())
    else:
        p = annotations_path(pool, base)
        if p.exists():
            p.unlink()


def clear_dialog(
    pool: str,
    dialog_id: str,
    base: Path = Path("data/annotations"),
) -> int:
    """Remove all verdicts for a (pool, dialog_id). Returns number removed."""
    current = load_pool(pool, base)
    removed = [k for k in current if k[0] == dialog_id]
    for k in removed:
        current.pop(k, None)
    if current:
        write_jsonl(annotations_path(pool, base), current.values())
    else:
        p = annotations_path(pool, base)
        if p.exists():
            p.unlink()
    return len(removed)
