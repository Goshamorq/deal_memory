"""Aggregate metrics from manual annotations.

Per (pool, field): accuracy = correct / annotated; partial-rate; wrong-rate.
Compared to per-field thresholds from Gate 2 D3 (interpreted as
"share of ✓ verdicts").
"""
from __future__ import annotations

from pydantic import BaseModel

from deal_memory.schema import FIELD_KEYS, Annotation

# Targets from Gate 2 D3 (originally F1, reused here as "share of ✓").
FIELD_TARGETS: dict[str, float] = {
    "budget": 0.85,
    "decision_maker": 0.75,
    "technical_requirements": 0.70,
    "objections": 0.65,
    "promises": 0.75,
    "next_step": 0.80,
}


class FieldStats(BaseModel):
    field: str
    correct: int
    partial: int
    wrong: int

    @property
    def total(self) -> int:
        return self.correct + self.partial + self.wrong

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def soft_accuracy(self) -> float:
        """Treat 'partial' as half-credit."""
        if not self.total:
            return 0.0
        return (self.correct + 0.5 * self.partial) / self.total

    @property
    def target(self) -> float:
        return FIELD_TARGETS.get(self.field, 0.0)

    @property
    def hits_target(self) -> bool:
        return self.accuracy >= self.target


class PoolReport(BaseModel):
    pool: str
    fields: list[FieldStats]
    n_annotations: int

    @property
    def macro_accuracy(self) -> float:
        scored = [f for f in self.fields if f.total > 0]
        if not scored:
            return 0.0
        return sum(f.accuracy for f in scored) / len(scored)

    @property
    def macro_soft_accuracy(self) -> float:
        scored = [f for f in self.fields if f.total > 0]
        if not scored:
            return 0.0
        return sum(f.soft_accuracy for f in scored) / len(scored)


def compute_pool_report(pool: str, annotations: dict[tuple[str, str], Annotation]) -> PoolReport:
    """Aggregate verdicts into per-field stats. annotations: keyed by (dialog_id, field)."""
    by_field: dict[str, dict[str, int]] = {
        f: {"correct": 0, "partial": 0, "wrong": 0} for f in FIELD_KEYS
    }
    for ann in annotations.values():
        if ann.field in by_field:
            by_field[ann.field][ann.verdict] += 1

    fields = [
        FieldStats(field=f, correct=c["correct"], partial=c["partial"], wrong=c["wrong"])
        for f, c in by_field.items()
    ]
    return PoolReport(pool=pool, fields=fields, n_annotations=len(annotations))
