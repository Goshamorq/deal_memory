"""LLM-driven extraction of DealFacts from a transcript.

Single repair attempt on schema-validation failure: we feed the validation
error back to the model and ask it to fix the JSON. Anything more elaborate
is out of scope for MVP-1.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from deal_memory.gigachat import LLMClient
from deal_memory.schema import DealFacts, Dialog, Prediction

EXTRACTION_TEMPERATURE = 0.1
PROMPT_PATH = Path("data/config/prompt.txt")


DEFAULT_SYSTEM_PROMPT = """Ты — система извлечения фактов из стенограмм B2B-продаж IT-оборудования.

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


def load_system_prompt(path: Path = PROMPT_PATH) -> str:
    """Read system prompt from file; fall back to DEFAULT_SYSTEM_PROMPT."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return DEFAULT_SYSTEM_PROMPT


def save_system_prompt(text: str, path: Path = PROMPT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_messages(
    transcript: str,
    system_prompt: str,
    repair_error: str | None = None,
) -> list[dict[str, str]]:
    user = f"Извлеки факты из транскрипта:\n\n{transcript}"
    if repair_error:
        user += (
            f"\n\nТвой предыдущий ответ не прошёл валидацию: {repair_error}. "
            "Верни тот же JSON, но исправь структуру."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def extract_one(
    client: LLMClient,
    sample_id: str,
    transcript: str,
    *,
    system_prompt: str | None = None,
) -> Prediction:
    """Extract DealFacts from one transcript. One repair attempt on schema fail."""
    prompt = system_prompt if system_prompt is not None else load_system_prompt()

    raw = client.chat(
        messages=_build_messages(transcript, prompt),
        temperature=EXTRACTION_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    repaired = False
    try:
        payload = _parse_payload(raw)
        facts = DealFacts.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raw = client.chat(
            messages=_build_messages(transcript, prompt, repair_error=str(exc)),
            temperature=EXTRACTION_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        repaired = True
        try:
            payload = _parse_payload(raw)
            facts = DealFacts.model_validate(payload)
        except (json.JSONDecodeError, ValidationError):
            return Prediction(
                id=sample_id, prediction=DealFacts(), raw_response=raw, parse_repaired=True
            )
    return Prediction(id=sample_id, prediction=facts, raw_response=raw, parse_repaired=repaired)


def extract_batch(
    client: LLMClient,
    dialogs: list[Dialog],
    *,
    system_prompt: str | None = None,
) -> list[Prediction]:
    """Run extract_one over a batch in order."""
    return [
        extract_one(client, d.id, d.transcript, system_prompt=system_prompt)
        for d in dialogs
    ]
