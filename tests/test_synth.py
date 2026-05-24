"""Tests for the synthetic generator."""
from __future__ import annotations

import pytest

from deal_memory import synth
from conftest import FakeLLMClient, make_synth_response


def test_generates_n_samples_with_seed():
    transcript = (
        "Менеджер: Добрый день, обсудим серверы 2U.\n"
        "Клиент: Да, нам нужно 12 серверов 2U, до 6 млн рублей.\n"
        "Менеджер: Пришлю спецификацию завтра до 18:00.\n"
        "Клиент: Технический директор согласует.\n"
    )
    response = make_synth_response(
        transcript=transcript,
        budget="до 6 млн рублей",
        decision_maker="Технический директор",
        promises=["Пришлю спецификацию завтра до 18:00"],
    )
    client = FakeLLMClient(responses=[response])

    samples = list(synth.generate(client, n=3, seed=42))
    assert len(samples) == 3
    for s in samples:
        assert s.transcript == transcript
        assert s.ground_truth.budget.value == "до 6 млн рублей"
        assert s.ground_truth.technical_requirements.value is None  # not revealed


def test_retries_when_validation_fails(monkeypatch):
    bad = make_synth_response(
        transcript="Менеджер: Привет.\nКлиент: Привет.",
        budget="100 миллионов рублей",  # not in transcript → will fail validation
    )
    good = make_synth_response(
        transcript="Менеджер: бюджет 6 млн рублей подходит?\nКлиент: да",
        budget="6 млн рублей",
    )
    client = FakeLLMClient(responses=[bad, good])

    samples = list(synth.generate(client, n=1, seed=1))
    assert len(samples) == 1
    assert samples[0].ground_truth.budget.value == "6 млн рублей"
    # one validation failure + one success = 2 calls
    assert len(client.calls) == 2


def test_raises_after_max_retries():
    bad = make_synth_response(
        transcript="Менеджер: Привет.\nКлиент: Привет.",
        budget="100 миллионов",
    )
    client = FakeLLMClient(responses=[bad])  # repeats forever
    with pytest.raises(RuntimeError, match="Generation failed"):
        list(synth.generate(client, n=1, seed=1))
    assert len(client.calls) == synth.MAX_GENERATION_RETRIES


def test_reveal_mask_sizes_between_2_and_6():
    import random

    for seed in range(20):
        rng = random.Random(seed)
        mask = synth._sample_reveal_mask(rng)
        assert 2 <= len(mask) <= 6
        assert mask.issubset(set(synth.FIELD_KEYS))
