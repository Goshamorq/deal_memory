# DealMemory MVP-1

Учебный thin-slice прототип вертикальной «памяти сделки» для B2B-продаж IT-оборудования.

Пайплайн:
1. Генерирует русскоязычные диалоги B2B IT-продаж с ground truth по 6 полям сделки.
2. Извлекает поля из транскриптов через GigaChat (LLM + JSON schema).
3. Считает F1 и hallucination rate против ground truth.
4. Streamlit-UI для просмотра диалогов, сравнения и сводных метрик.

## Setup

```bash
uv sync --all-extras
cp .env.example .env
# заполнить GIGACHAT_AUTH_KEY в .env
```

## Команды

```bash
uv run dm synth generate --n 50            # 50 синтетических диалогов → data/synthetic/v1.jsonl
uv run dm extract run --in data/synthetic/v1.jsonl
uv run dm eval score --predictions data/eval-runs/<ts>.jsonl --truth data/synthetic/v1.jsonl
uv run dm ui                                # Streamlit на :8501
uv run pytest -q                            # юнит-тесты с FakeLLMClient
```

## Поля сделки

| Поле | Описание |
|---|---|
| `budget` | Заявленный или обсуждаемый бюджет |
| `decision_maker` | Кто принимает / согласует / блокирует решение |
| `technical_requirements` | Спецификации, интеграции, сертификации |
| `objections` | Незакрытые возражения клиента (список) |
| `promises` | Обещания менеджера со сроками (список) |
| `next_step` | Следующий шаг: дата + тип контакта |

## Стек

Python 3.11+, pydantic, httpx, typer, streamlit, pytest. Менеджер пакетов — `uv`.

## Структура

```
src/deal_memory/
├── schema.py      # Pydantic-модели
├── gigachat.py    # клиент GigaChat (OAuth + chat)
├── synth.py       # генератор синтетики
├── extract.py     # экстрактор 6 полей
├── eval.py        # метрики и runner
├── storage.py     # JSONL I/O
├── ui.py          # Streamlit (3 страницы)
└── cli.py         # typer-команды
```

Полный план — [plan-mvp1.md](docs/roadmap.md) и Gate 2 PDF.
