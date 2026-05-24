# DealMemory

Учебный thin-slice прототип вертикальной «памяти сделки» для B2B-продаж IT-оборудования.

## Что делает

1. Хранит **пулы диалогов** в JSONL (`data/pools/`) — телефонные звонки и email-цепочки B2B IT-продаж.
2. По кнопке **«Обработать»** прогоняет диалог через GigaChat и извлекает 6 ключевых полей сделки в JSON (с repair-pass на ошибку схемы).
3. Под каждым полем — три кнопки **✗ ± ✓** для ручной оценки качества извлечения. Разметка живёт в `data/annotations/`.
4. Вкладка «Метрики» агрегирует разметку и сравнивает с таргетами accuracy из Gate 2.
5. Вкладка «Настройки» — редактор system prompt с применением на лету.

## Setup

См. [QUICKSTART.md](QUICKSTART.md) — 8 шагов от `git clone` до запущенного UI.

## 6 полей сделки

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
├── schema.py        # Pydantic-модели (Dialog, DealFacts, Prediction, Annotation)
├── gigachat.py      # клиент GigaChat (OAuth + chat, JSON-mode)
├── extract.py       # экстрактор 6 полей + load/save system prompt
├── annotations.py   # хранилище ручной разметки (JSONL per-pool)
├── metrics.py       # агрегация ✗/±/✓ → per-field accuracy + таргеты
├── storage.py       # JSONL I/O
├── ui.py            # Streamlit (3 вкладки + wizard)
└── cli.py           # typer: dm extract run | dm ui
```

Данные:

```
data/
├── pools/<pool>.jsonl          # ✓ в git — диалоги
├── eval-runs/<pool>.jsonl      # gitignored — predictions GigaChat
├── annotations/<pool>.jsonl    # gitignored — ✗/±/✓ разметка
└── config/prompt.txt           # ✓ в git — текущий system prompt
```

## Тесты

```bash
uv run pytest -q     # 15 тестов с FakeLLMClient, без сети
```

## Дополнительно

- [CLAUDE.md](CLAUDE.md) — guidelines для агентного кодинга (Karpathy-style)
- [docs/roadmap.md](docs/roadmap.md) — что после MVP-1 (STT, Bitrix24, risk scoring)
