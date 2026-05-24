"""Tests for the LLM extractor."""
from __future__ import annotations

import json

from conftest import FakeLLMClient

from deal_memory import extract


def _valid_facts_json(*, budget=None, decision_maker=None, objections=None) -> str:
    def field(v):
        return {"value": v, "quote": v, "confidence": 1.0 if v else 0.0}

    return json.dumps(
        {
            "budget": field(budget),
            "decision_maker": field(decision_maker),
            "technical_requirements": field(None),
            "objections": [
                {"value": o, "quote": o, "confidence": 0.9} for o in (objections or [])
            ],
            "promises": [],
            "next_step": field(None),
        }
    )


def test_extract_returns_validated_prediction():
    client = FakeLLMClient(responses=[_valid_facts_json(budget="5 млн ₽")])

    pred = extract.extract_one(client, "abc", "Менеджер: ... Клиент: ...")

    assert pred.id == "abc"
    assert pred.prediction.budget.value == "5 млн ₽"
    assert pred.prediction.decision_maker.value is None
    assert pred.parse_repaired is False
    assert len(client.calls) == 1


def test_extract_repairs_on_invalid_json():
    broken = "not a json"
    fixed = _valid_facts_json(budget="3 млн ₽")
    client = FakeLLMClient(responses=[broken, fixed])

    pred = extract.extract_one(client, "id1", "transcript")

    assert pred.prediction.budget.value == "3 млн ₽"
    assert pred.parse_repaired is True
    assert len(client.calls) == 2
    # repair message must include the original error mention
    repair_user_msg = client.calls[1]["messages"][1]["content"]
    assert "не прошёл валидацию" in repair_user_msg


def test_extract_returns_empty_facts_when_repair_also_fails():
    client = FakeLLMClient(responses=["broken 1", "still broken"])

    pred = extract.extract_one(client, "id2", "transcript")

    # Empty facts: all FieldEvidence default to value=None
    assert pred.prediction.budget.value is None
    assert pred.prediction.objections == []
    assert pred.parse_repaired is True
    assert pred.raw_response == "still broken"


def test_extract_strips_markdown_fences():
    payload = _valid_facts_json(budget="1 млн ₽")
    fenced = f"```json\n{payload}\n```"
    client = FakeLLMClient(responses=[fenced])

    pred = extract.extract_one(client, "id3", "transcript")
    assert pred.prediction.budget.value == "1 млн ₽"
    assert pred.parse_repaired is False


def test_load_model_name_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.delenv("GIGACHAT_MODEL", raising=False)
    assert extract.load_model_name(tmp_path / "missing.txt") == extract.DEFAULT_MODEL


def test_save_then_load_model_name(tmp_path):
    p = tmp_path / "model.txt"
    extract.save_model_name("GigaChat-2-Pro", p)
    assert extract.load_model_name(p) == "GigaChat-2-Pro"


def test_save_model_name_rejects_unknown(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="Unknown model"):
        extract.save_model_name("InventedModel", tmp_path / "model.txt")


def test_load_model_name_ignores_garbage_in_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GIGACHAT_MODEL", raising=False)
    p = tmp_path / "model.txt"
    p.write_text("not-a-real-model", encoding="utf-8")
    assert extract.load_model_name(p) == extract.DEFAULT_MODEL


def test_load_model_name_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGACHAT_MODEL", "GigaChat-2-Max")
    assert extract.load_model_name(tmp_path / "missing.txt") == "GigaChat-2-Max"
