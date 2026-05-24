"""Synthetic dialog generator for the B2B IT-equipment sales vertical.

Each generation produces a dialog + ground-truth labels for the 6 DealFacts
fields. Critically, for each dialog we pre-decide which fields are revealed:
this lets the eval set include the "field not mentioned" case, which is
required to measure hallucination rate honestly (Gate 2 D4 risk #2).

Two-step generation:
  1. Ask the LLM for transcript + ground_truth as a single JSON object.
  2. Validate that every non-null ground-truth value is actually quoted in
     the transcript (case-insensitive substring). Retry up to 3 times.
"""
from __future__ import annotations

import json
import random
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from deal_memory.gigachat import LLMClient
from deal_memory.schema import DealFacts, Scenario, SyntheticSample

MAX_GENERATION_RETRIES = 3
GENERATION_TEMPERATURE = 0.8

FIELD_KEYS = (
    "budget",
    "decision_maker",
    "technical_requirements",
    "objections",
    "promises",
    "next_step",
)


@dataclass(frozen=True)
class Persona:
    id: str
    company: str
    industry: str
    description: str


PERSONAS: tuple[Persona, ...] = (
    Persona(
        id="promstroy",
        company="ООО «Промстрой»",
        industry="строительство, генподрядчик",
        description=(
            "крупный генподрядчик, обновляют серверный парк под 1С и видеонаблюдение, "
            "ищут серверы 2U с дисками NVMe, чувствительны к срокам и гарантии"
        ),
    ),
    Persona(
        id="technoservice",
        company="ООО «ТехноСервис»",
        industry="логистика, оператор СВХ",
        description=(
            "оператор склада временного хранения, обновляют СХД для системы учёта, "
            "сравнивают с конкурентами по цене, требуют расчёт TCO"
        ),
    ),
    Persona(
        id="energoholding",
        company="АО «Энерго-Холдинг»",
        industry="энергетика, IIoT",
        description=(
            "энергетическая компания, разворачивают IIoT-камеры на подстанциях, "
            "нужна сертификация Минпромторга и поддержка протоколов MQTT/Modbus"
        ),
    ),
    Persona(
        id="transmash",
        company="АО «ТрансМаш»",
        industry="промышленность, машиностроение",
        description=(
            "машиностроительный завод, заменяют сетевое оборудование цеха, "
            "критичны промышленные коммутаторы и резервирование"
        ),
    ),
    Persona(
        id="stalprom",
        company="ПК «СтальПром»",
        industry="металлургия",
        description=(
            "металлургический комбинат, строят серверный кластер для MES-системы, "
            "длинная цепочка согласований, важна импортонезависимость"
        ),
    ),
    Persona(
        id="datacenter_nsk",
        company="«Дата-Центр НСК»",
        industry="дата-центр, colocation",
        description=(
            "региональный ЦОД в Новосибирске, расширяют пул сетевого оборудования, "
            "требуют 10GbE/25GbE и поддержку BGP"
        ),
    ),
    Persona(
        id="medlab",
        company="ООО «МедЛаб»",
        industry="медицина, лабораторная диагностика",
        description=(
            "сеть медицинских лабораторий, обновляют серверы для LIS-системы, "
            "требования ФЗ-152 по хранению ПДн пациентов"
        ),
    ),
    Persona(
        id="agroprom",
        company="АО «АгроПром»",
        industry="агропром, тепличное хозяйство",
        description=(
            "агрохолдинг, IIoT-датчики микроклимата теплиц, "
            "ограниченный бюджет, требуют пилот перед заказом всей партии"
        ),
    ),
)


@dataclass(frozen=True)
class ScenarioSpec:
    key: Scenario
    description: str


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        key="cold_call",
        description="первый холодный звонок менеджера IT-интегратора потенциальному клиенту, разведка ситуации и квалификация",
    ),
    ScenarioSpec(
        key="demo",
        description="демонстрация решения с участием технического директора со стороны клиента",
    ),
    ScenarioSpec(
        key="spec_alignment",
        description="согласование технических требований и конфигурации оборудования",
    ),
    ScenarioSpec(
        key="budget_discussion",
        description="обсуждение бюджета, скидок и условий оплаты с финансовым директором",
    ),
)


SYSTEM_PROMPT = """Ты — генератор реалистичных русскоязычных диалогов B2B-продаж IT-оборудования.
Твоя задача — создавать обучающие диалоги для системы экстракции ключевых фактов сделки.

Главное правило: если поле помечено как "не раскрывать" — ОНО НЕ ДОЛЖНО упоминаться в диалоге даже косвенно.
Если поле помечено как "раскрыть" — оно должно быть проговорено явно, его конкретное значение должно встречаться в транскрипте дословно (или близко к дословному) так, чтобы потом можно было сделать substring-проверку.

Возвращай строго JSON по схеме:
{
  "transcript": "Менеджер: ...\\nКлиент: ...\\n... (10-25 реплик)",
  "ground_truth": {
    "budget": {"value": "..." | null, "quote": "точная цитата из transcript" | null, "confidence": 1.0},
    "decision_maker": {"value": "..." | null, "quote": "..." | null, "confidence": 1.0},
    "technical_requirements": {"value": "..." | null, "quote": "..." | null, "confidence": 1.0},
    "objections": [{"value": "...", "quote": "...", "confidence": 1.0}, ...],
    "promises": [{"value": "...", "quote": "...", "confidence": 1.0}, ...],
    "next_step": {"value": "..." | null, "quote": "..." | null, "confidence": 1.0}
  }
}

Для "не раскрытого" поля используй value=null, quote=null, confidence=0.0.
Для списков (objections, promises): если поле "не раскрыто" — верни пустой список [].
"""


FIELD_DESCRIPTIONS = {
    "budget": "заявленный или ориентировочный бюджет сделки",
    "decision_maker": "ЛПР — кто принимает решение и кто согласует",
    "technical_requirements": "технические требования: модели, спецификации, интеграции, сертификации",
    "objections": "незакрытые возражения клиента (1-3 шт): цена, сроки, гарантия, поддержка и т.п.",
    "promises": "обещания менеджера с указанием срока (1-3 шт)",
    "next_step": "следующий шаг: тип контакта + дата/срок",
}


def _build_user_prompt(persona: Persona, scenario: ScenarioSpec, reveal: set[str]) -> str:
    reveal_block = "\n".join(
        f"  - {k}: {FIELD_DESCRIPTIONS[k]}" for k in FIELD_KEYS if k in reveal
    )
    hide_block = "\n".join(f"  - {k}" for k in FIELD_KEYS if k not in reveal)
    return f"""Сгенерируй диалог для сценария: {scenario.description}.

Клиент: {persona.company} ({persona.industry}). Контекст: {persona.description}.

Раскрыть в диалоге следующие поля:
{reveal_block or '  (нет полей для раскрытия — диалог должен оборваться рано)'}

НЕ упоминать (даже косвенно):
{hide_block or '  (все поля разрешены)'}

Диалог должен звучать естественно, 10-25 реплик. Реплики чередуются «Менеджер:» / «Клиент:» (можно ввести «Технический директор клиента:» или «ФД клиента:» если это релевантно сценарию).
Верни только JSON, без пояснений."""


def _sample_reveal_mask(rng: random.Random) -> set[str]:
    """Pick 2-6 fields to reveal in this dialog."""
    n = rng.randint(2, 6)
    return set(rng.sample(FIELD_KEYS, n))


def _validate_transcript_coverage(sample: SyntheticSample) -> list[str]:
    """Return list of error strings; empty list = valid."""
    errors: list[str] = []
    transcript_norm = sample.transcript.lower()

    def check(field_name: str, value: str | None, quote: str | None) -> None:
        if value is None:
            return
        # check value substring (loose: first ~20 chars after normalization)
        needle = value.lower().strip()
        if needle and needle[: min(20, len(needle))] not in transcript_norm:
            errors.append(f"{field_name}.value not found in transcript: {value!r}")
        if quote and quote.lower().strip()[:20] not in transcript_norm:
            errors.append(f"{field_name}.quote not found in transcript: {quote!r}")

    gt = sample.ground_truth
    check("budget", gt.budget.value, gt.budget.quote)
    check("decision_maker", gt.decision_maker.value, gt.decision_maker.quote)
    check("technical_requirements", gt.technical_requirements.value, gt.technical_requirements.quote)
    check("next_step", gt.next_step.value, gt.next_step.quote)
    for i, item in enumerate(gt.objections):
        check(f"objections[{i}]", item.value, item.quote)
    for i, item in enumerate(gt.promises):
        check(f"promises[{i}]", item.value, item.quote)
    return errors


def _parse_llm_payload(raw: str) -> dict[str, Any]:
    """LLMs sometimes wrap JSON in ```json fences — strip them."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # drop optional 'json' language tag on first line
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    return json.loads(text)


def _generate_one(
    client: LLMClient,
    persona: Persona,
    scenario: ScenarioSpec,
    reveal: set[str],
) -> SyntheticSample:
    last_error: str | None = None
    for attempt in range(1, MAX_GENERATION_RETRIES + 1):
        prompt = _build_user_prompt(persona, scenario, reveal)
        if last_error:
            prompt += f"\n\nПредыдущая попытка не прошла валидацию: {last_error}. Исправь."
        raw = client.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=GENERATION_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        try:
            payload = _parse_llm_payload(raw)
            sample = SyntheticSample(
                id=str(uuid.uuid4()),
                scenario=scenario.key,
                transcript=payload["transcript"],
                ground_truth=DealFacts.model_validate(payload["ground_truth"]),
                meta={"persona_id": persona.id, "reveal": sorted(reveal), "attempt": attempt},
            )
        except (json.JSONDecodeError, ValidationError, KeyError) as exc:
            last_error = f"parse/schema error: {exc}"
            continue
        errors = _validate_transcript_coverage(sample)
        if not errors:
            return sample
        last_error = "; ".join(errors)
    raise RuntimeError(
        f"Generation failed after {MAX_GENERATION_RETRIES} attempts. Last error: {last_error}"
    )


def generate(client: LLMClient, n: int, seed: int | None = None) -> Iterator[SyntheticSample]:
    """Yield n validated SyntheticSamples."""
    rng = random.Random(seed)
    for _ in range(n):
        persona = rng.choice(PERSONAS)
        scenario = rng.choice(SCENARIOS)
        reveal = _sample_reveal_mask(rng)
        yield _generate_one(client, persona, scenario, reveal)
