"""Tests for the eval harness."""
from __future__ import annotations

from deal_memory import eval as eval_mod
from deal_memory.schema import DealFacts, FieldEvidence, Prediction, SyntheticSample


def _fe(v: str | None = None) -> FieldEvidence:
    return FieldEvidence(value=v, quote=v, confidence=1.0 if v else 0.0)


def _sample(sid: str, **kwargs) -> SyntheticSample:
    facts = DealFacts(**{k: v for k, v in kwargs.items()})
    return SyntheticSample(id=sid, scenario="demo", transcript="...", ground_truth=facts)


def _pred(sid: str, **kwargs) -> Prediction:
    facts = DealFacts(**{k: v for k, v in kwargs.items()})
    return Prediction(id=sid, prediction=facts)


def test_jaccard_basic():
    assert eval_mod.jaccard("Технический директор", "технический Директор") == 1.0
    assert eval_mod.jaccard("", "") == 1.0
    assert eval_mod.jaccard("a", "") == 0.0
    assert 0.0 < eval_mod.jaccard("серверы 2U NVMe", "серверы 2U") < 1.0


def test_scalar_field_tp_and_fn_and_fp():
    truth = [
        _sample("1", budget=_fe("5 млн рублей")),
        _sample("2", budget=_fe("3 млн рублей")),
        _sample("3", budget=_fe(None)),
    ]
    preds = [
        _pred("1", budget=_fe("5 млн рублей")),         # TP
        _pred("2", budget=_fe(None)),                    # FN
        _pred("3", budget=_fe("1 млн рублей")),          # FP (hallucination)
    ]
    report = eval_mod.score(truth, preds)
    budget_metrics = next(f for f in report.fields if f.field == "budget")
    assert budget_metrics.tp == 1
    assert budget_metrics.fn == 1
    assert budget_metrics.fp == 1


def test_hallucination_rate_and_coverage():
    truth = [
        _sample("1", budget=_fe(None), decision_maker=_fe("Технический директор")),
        _sample("2", budget=_fe(None), decision_maker=_fe(None)),
    ]
    preds = [
        # halluc on budget, correct on DM
        _pred("1", budget=_fe("5 млн"), decision_maker=_fe("технический директор")),
        # no hallucination
        _pred("2", budget=_fe(None), decision_maker=_fe(None)),
    ]
    report = eval_mod.score(truth, preds)
    # 4 scalar fields × 2 samples - truth had values 1, none 7
    # hallucinated 1 (sample 1, budget); halluc total 7
    assert report.hallucination_count == 1
    assert report.hallucination_total == 7
    assert report.hallucination_rate == 1 / 7
    # coverage: 1 truth value, correctly predicted
    assert report.coverage_correct == 1
    assert report.coverage_total == 1


def test_list_field_greedy_match():
    truth = [
        _sample(
            "1",
            objections=[_fe("дорого"), _fe("долгие сроки поставки")],
        )
    ]
    preds = [
        _pred(
            "1",
            objections=[_fe("долгие сроки"), _fe("опыт интегратора неизвестен")],
        )
    ]
    report = eval_mod.score(truth, preds)
    obj_metrics = next(f for f in report.fields if f.field == "objections")
    # "долгие сроки" jaccard with "долгие сроки поставки" = 2/3 > 0.5 -> TP
    # "дорого" unmatched -> FN
    # "опыт интегратора неизвестен" unmatched -> FP
    assert obj_metrics.tp == 1
    assert obj_metrics.fn == 1
    assert obj_metrics.fp == 1


def test_macro_f1_average():
    truth = [_sample("1", budget=_fe("5 млн"), next_step=_fe("звонок 21.03"))]
    preds = [_pred("1", budget=_fe("5 млн"), next_step=_fe("звонок 21.03"))]
    report = eval_mod.score(truth, preds)
    # 4 scalar + 2 list = 6 fields; budget+next_step F1=1.0, others 0.0
    assert 0.3 < report.macro_f1 < 0.4
