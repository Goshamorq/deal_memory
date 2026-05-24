"""Tests for the annotation-based metrics aggregator."""
from __future__ import annotations

from deal_memory import metrics
from deal_memory.schema import Annotation


def _ann(did: str, field: str, verdict: str) -> Annotation:
    return Annotation(dialog_id=did, field=field, verdict=verdict)


def test_per_field_counts_and_accuracy():
    anns = {
        ("d1", "budget"): _ann("d1", "budget", "correct"),
        ("d2", "budget"): _ann("d2", "budget", "correct"),
        ("d3", "budget"): _ann("d3", "budget", "wrong"),
        ("d4", "budget"): _ann("d4", "budget", "partial"),
    }
    report = metrics.compute_pool_report("p", anns)

    budget = next(f for f in report.fields if f.field == "budget")
    assert budget.correct == 2
    assert budget.partial == 1
    assert budget.wrong == 1
    assert budget.total == 4
    assert budget.accuracy == 0.5
    assert budget.soft_accuracy == (2 + 0.5) / 4


def test_macro_accuracy_ignores_empty_fields():
    anns = {("d1", "budget"): _ann("d1", "budget", "correct")}
    report = metrics.compute_pool_report("p", anns)
    # Only budget has any data → macro = 1.0
    assert report.macro_accuracy == 1.0


def test_hits_target_flag():
    # budget target is 0.85 → 1 correct out of 1 = 1.0 > 0.85 → hit
    anns = {("d1", "budget"): _ann("d1", "budget", "correct")}
    report = metrics.compute_pool_report("p", anns)
    budget = next(f for f in report.fields if f.field == "budget")
    assert budget.hits_target is True


def test_n_annotations():
    anns = {
        ("d1", "budget"): _ann("d1", "budget", "correct"),
        ("d1", "decision_maker"): _ann("d1", "decision_maker", "wrong"),
    }
    report = metrics.compute_pool_report("p", anns)
    assert report.n_annotations == 2
