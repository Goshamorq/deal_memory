"""Eval harness — F1 per field + hallucination rate.

Match strategy is Jaccard on lowercased word tokens, threshold 0.5 by
default. Same threshold for scalar and per-item list matches; we keep it
uniform on purpose (Karpathy: simpler == better until evidence demands
field-specific normalizers).
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel

from deal_memory.schema import DealFacts, FieldEvidence, Prediction, SyntheticSample

JACCARD_THRESHOLD = 0.5
SCALAR_FIELDS = ("budget", "decision_maker", "technical_requirements", "next_step")
LIST_FIELDS = ("objections", "promises")
ALL_FIELDS = SCALAR_FIELDS + LIST_FIELDS

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


class FieldMetrics(BaseModel):
    field: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


class EvalReport(BaseModel):
    n_samples: int
    fields: list[FieldMetrics]
    hallucination_count: int
    hallucination_total: int  # number of (truth.value is None) slots
    coverage_correct: int
    coverage_total: int  # number of (truth.value is not None) slots

    @property
    def hallucination_rate(self) -> float:
        return self.hallucination_count / self.hallucination_total if self.hallucination_total else 0.0

    @property
    def coverage(self) -> float:
        return self.coverage_correct / self.coverage_total if self.coverage_total else 0.0

    @property
    def macro_f1(self) -> float:
        if not self.fields:
            return 0.0
        return sum(f.f1 for f in self.fields) / len(self.fields)


# ---- Tokenisation / match primitives ----


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _values_match(truth: str, pred: str, threshold: float = JACCARD_THRESHOLD) -> bool:
    return jaccard(truth, pred) >= threshold


# ---- Per-field counting ----


def _count_scalar(
    field: str,
    truths: Iterable[FieldEvidence],
    preds: Iterable[FieldEvidence],
    threshold: float,
) -> FieldMetrics:
    tp = fp = fn = 0
    for t, p in zip(truths, preds, strict=True):
        t_has = t.value is not None
        p_has = p.value is not None
        if t_has and p_has:
            if _values_match(t.value or "", p.value or "", threshold):
                tp += 1
            else:
                fp += 1  # wrong value AND missed truth → count as FP only
                fn += 1
        elif t_has and not p_has:
            fn += 1
        elif p_has and not t_has:
            fp += 1
    return FieldMetrics(field=field, tp=tp, fp=fp, fn=fn)


def _greedy_match_lists(
    truth_items: list[FieldEvidence],
    pred_items: list[FieldEvidence],
    threshold: float,
) -> tuple[int, int, int]:
    """Greedy 1-to-1 matching by max Jaccard."""
    used_pred: set[int] = set()
    tp = 0
    for t in truth_items:
        best_idx = -1
        best_score = -1.0
        for i, p in enumerate(pred_items):
            if i in used_pred:
                continue
            score = jaccard(t.value or "", p.value or "")
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx >= 0 and best_score >= threshold:
            used_pred.add(best_idx)
            tp += 1
    fn = len(truth_items) - tp
    fp = len(pred_items) - len(used_pred)
    return tp, fp, fn


def _count_list(
    field: str,
    truths: Iterable[list[FieldEvidence]],
    preds: Iterable[list[FieldEvidence]],
    threshold: float,
) -> FieldMetrics:
    tp = fp = fn = 0
    for t_items, p_items in zip(truths, preds, strict=True):
        a, b, c = _greedy_match_lists(t_items, p_items, threshold)
        tp += a
        fp += b
        fn += c
    return FieldMetrics(field=field, tp=tp, fp=fp, fn=fn)


# ---- Top-level scoring ----


def _field_attr(facts: DealFacts, name: str):
    return getattr(facts, name)


def score(
    truth: list[SyntheticSample],
    predictions: list[Prediction],
    threshold: float = JACCARD_THRESHOLD,
) -> EvalReport:
    """Match by id, compute per-field F1 + hallucination/coverage."""
    pred_by_id = {p.id: p.prediction for p in predictions}
    paired: list[tuple[DealFacts, DealFacts]] = []
    for sample in truth:
        if sample.id in pred_by_id:
            paired.append((sample.ground_truth, pred_by_id[sample.id]))
    if not paired:
        raise ValueError("No (truth, prediction) pairs matched by id.")

    fields_metrics: list[FieldMetrics] = []
    for f in SCALAR_FIELDS:
        fields_metrics.append(
            _count_scalar(
                f,
                [_field_attr(g, f) for g, _ in paired],
                [_field_attr(p, f) for _, p in paired],
                threshold,
            )
        )
    for f in LIST_FIELDS:
        fields_metrics.append(
            _count_list(
                f,
                [_field_attr(g, f) for g, _ in paired],
                [_field_attr(p, f) for _, p in paired],
                threshold,
            )
        )

    # Hallucination rate (only scalar fields: list fields collapse to count via FP)
    hall_count = 0
    hall_total = 0
    cov_correct = 0
    cov_total = 0
    for g, p in paired:
        for f in SCALAR_FIELDS:
            t_fe: FieldEvidence = _field_attr(g, f)
            p_fe: FieldEvidence = _field_attr(p, f)
            if t_fe.value is None:
                hall_total += 1
                if p_fe.value is not None:
                    hall_count += 1
            else:
                cov_total += 1
                if p_fe.value is not None and _values_match(t_fe.value, p_fe.value, threshold):
                    cov_correct += 1

    return EvalReport(
        n_samples=len(paired),
        fields=fields_metrics,
        hallucination_count=hall_count,
        hallucination_total=hall_total,
        coverage_correct=cov_correct,
        coverage_total=cov_total,
    )
