"""LLM-driven extraction of DealFacts from a transcript.

Single repair attempt on schema-validation failure: we feed the validation
error back to the model and ask it to fix the JSON. Anything more elaborate
is out of scope for MVP-1.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from deal_memory.gigachat import LLMClient
from deal_memory.schema import DealFacts, Prediction, SyntheticSample

EXTRACTION_TEMPERATURE = 0.1


SYSTEM_PROMPT = """Ты — система извлечения фактов из стенограмм B2B-продаж IT-оборудования.

Тебе на вход дают транскрипт телефонного звонка между менеджером IT-интегратора и клиентом. Ты извлекаешь 6 структурированных полей сделки:

1. budget — заявленный или обсуждаемый бюджет сделки (значение + единица: «5 млн ₽», «до 200 тыс ₽» и т.п.)
2. decision_maker — ЛПР на стороне клиента: должность + (при наличии) имя
3. technical_requirements — конкретные технические требования: модели, спецификации, протоколы, сертификации (NVMe, 10GbE, MQTT, Минпромторг, ФЗ-152 и т.п.)
4. objections — список незакрытых возражений клиента (цена, сроки, гарантия, поддержка, опыт интегратора)
5. promises — список обещаний менеджера со сроком («Пришлю спецификацию завтра до 18:00»)
6. next_step — следующий шаг: тип контакта + дата/срок («Демо для техдиректора 21.03»)

КРИТИЧЕСКОЕ ПРАВИЛО: если факт НЕ был озвучен в транскрипте — значение поля = null. НЕ выдумывай. НЕ заполняй null-поля «по смыслу». Если поле упомянуто косвенно и неоднозначно — null.

Для каждого извлечённого факта обязательно укажи `quote` — короткую дословную цитату (5-15 слов) из транскрипта, подтверждающую факт.

Возвращай строго JSON по схеме:
{
  "budget": {"value": "..." | null, "quote": "..." | null, "confidence": 0..1},
  "decision_maker": {"value": "..." | null, "quote": "..." | null, "confidence": 0..1},
  "technical_requirements": {"value": "..." | null, "quote": "..." | null, "confidence": 0..1},
  "objections": [{"value": "...", "quote": "...", "confidence": 0..1}, ...],
  "promises": [{"value": "...", "quote": "...", "confidence": 0..1}, ...],
  "next_step": {"value": "..." | null, "quote": "..." | null, "confidence": 0..1}
}

Для списков objections и promises: если возражений/обещаний не было — верни пустой массив [].
confidence — твоя субъективная уверенность 0..1.
"""


def _parse_payload(raw: str) -> dict[str, Any]:
    """Strip optional ```json fences and parse."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    return json.loads(text)


def _build_messages(transcript: str, repair_error: str | None = None) -> list[dict[str, str]]:
    user = f"Извлеки факты из транскрипта:\n\n{transcript}"
    if repair_error:
        user += (
            f"\n\nТвой предыдущий ответ не прошёл валидацию: {repair_error}. "
            "Верни тот же JSON, но исправь структуру."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_one(client: LLMClient, sample_id: str, transcript: str) -> Prediction:
    """Extract DealFacts from one transcript. One repair attempt on schema fail."""
    raw = client.chat(
        messages=_build_messages(transcript),
        temperature=EXTRACTION_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    repaired = False
    try:
        payload = _parse_payload(raw)
        facts = DealFacts.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        # One repair pass
        raw = client.chat(
            messages=_build_messages(transcript, repair_error=str(exc)),
            temperature=EXTRACTION_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        repaired = True
        try:
            payload = _parse_payload(raw)
            facts = DealFacts.model_validate(payload)
        except (json.JSONDecodeError, ValidationError):
            # Give up: return empty facts with the raw response saved.
            return Prediction(
                id=sample_id, prediction=DealFacts(), raw_response=raw, parse_repaired=True
            )
    return Prediction(id=sample_id, prediction=facts, raw_response=raw, parse_repaired=repaired)


def extract_batch(client: LLMClient, samples: list[SyntheticSample]) -> list[Prediction]:
    """Run extract_one over a batch in order."""
    return [extract_one(client, s.id, s.transcript) for s in samples]
