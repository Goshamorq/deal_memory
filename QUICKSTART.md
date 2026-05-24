# Quickstart

Поднять локально UI с разметкой диалогов через GigaChat — за ~10 минут.

## 1. Что нужно поставить заранее

- **Python 3.11+** (`python3 --version`)
- **uv** — менеджер пакетов: [инструкция](https://docs.astral.sh/uv/getting-started/installation/) или `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git** (`git --version`)
- **GigaChat API key** — получи бесплатный ключ на [developers.sber.ru/portal/products/gigachat-api](https://developers.sber.ru/portal/products/gigachat-api): зарегистрируйся → создай проект → скопируй **Authorization key** (это base64 `client_id:secret`)

## 2. Клонирование и установка

```bash
git clone https://github.com/Goshamorq/deal_memory.git
cd deal_memory
uv sync --all-extras
```

`uv sync` создаст `.venv/`, поставит зависимости из `pyproject.toml` + `uv.lock` (детерминированно).

## 3. Конфигурация GigaChat

```bash
cp .env.example .env
```

Открой `.env` и заполни:

```env
GIGACHAT_AUTH_KEY=ВСТАВЬ_СВОЙ_AUTH_KEY_СЮДА
GIGACHAT_SCOPE=GIGACHAT_API_PERS      # для личного использования; для юрлица — GIGACHAT_API_B2B
GIGACHAT_MODEL=GigaChat                # или GigaChat-Pro / GigaChat-Max если есть доступ
GIGACHAT_VERIFY_SSL=false              # true только если установлен Russian Trusted Root CA
```

`.env` в `.gitignore` — секреты в репо не утекут.

## 4. Запуск UI

```bash
uv run dm ui
```

Откроется на [http://localhost:8501](http://localhost:8501). При первом запуске поднимется **wizard из 4 шагов** — пройди его, он объяснит навигацию. Перезапустить wizard можно фиолетовой кнопкой `🪄 Wizard` справа сверху.

## 5. Что делать в UI

**Вкладка «Диалоги»**:

1. Сверху выбери пул в первом dropdown'е (по умолчанию `manual-v1` — 5 коротких звонков; есть и `manual-big` — 3 длинных email-цепочки)
2. Выбери конкретный диалог во втором dropdown'е (`·` = ещё не обработан, `✓` = есть предсказание)
3. Слева — текст диалога в виде чата. Справа — 6 пустых полей сделки
4. Нажми **«Обработать»** — GigaChat за 5-15 сек заполнит поля
5. Под каждым полем три кнопки **✗ ± ✓** — поставь оценку качеству извлечения. Сохраняется сразу
6. «Очистить предсказание» — удалить prediction и обработать заново

**Вкладка «Метрики»**: сводная статистика по разметке выбранного пула. Macro accuracy, soft accuracy (где ± = 0.5), per-field таблица с попаданием в таргеты, bar chart распределения ✗/±/✓.

**Вкладка «Настройки»**: редактор system prompt. Меняй текст → «Сохранить» → новый prompt применится при следующем «Обработать» (без рестарта). Тут же env и таргеты.

## 6. Где что лежит

```
data/
├── pools/                  # ✓ в git — справочные диалоги по пулам
│   ├── manual-v1.jsonl     # 5 эталонных диалогов (звонки)
│   └── manual-big.jsonl    # 3 длинных email-диалога
├── eval-runs/              # ✗ gitignored — предсказания GigaChat, кешируются по пулу
│   └── <пул>.jsonl
├── annotations/            # ✗ gitignored — твоя ✗/±/✓ разметка
│   └── <пул>.jsonl
└── config/
    └── prompt.txt          # ✓ в git — текущий system prompt
```

Каждый файл — JSONL (одна Pydantic-модель на строку). Чтение/запись — через `deal_memory.storage`.

## 7. Юнит-тесты

```bash
uv run pytest -q
```

12 тестов, все используют `FakeLLMClient` (нет сетевых вызовов). Покрывают: extract repair-pass, annotations storage, metrics aggregation.

## 8. Добавить свой пул

1. Создай `data/pools/my-pool.jsonl` с диалогами в формате:
   ```jsonl
   {"id": "uniq-1", "scenario": "demo", "transcript": "Менеджер: ...\nКлиент: ...", "meta": {}}
   ```
   `scenario` — один из `cold_call | demo | spec_alignment | budget_discussion | handover`.  
   `transcript` — текст с маркерами `Менеджер:` / `Клиент:` / `Новый менеджер:` / `ФД клиента:` / `Финансовый директор:` в начале каждой реплики.
2. Перезагрузи страницу — пул появится в dropdown'е автоматически.

## 9. Batch-обработка через CLI (опционально)

Вместо одиночного клика «Обработать» в UI можно прогнать весь пул разом:

```bash
uv run dm extract run --in data/pools/manual-big.jsonl
# → data/eval-runs/manual-big.jsonl
```

UI подхватит результаты сразу, без рестарта.

## Troubleshooting

| Симптом | Решение |
|---|---|
| `GIGACHAT_AUTH_KEY is not set` | `.env` не загружен или ключ пуст. Проверь, что файл существует в корне проекта и заполнен |
| `SSL handshake failed` / `CERTIFICATE_VERIFY_FAILED` | В `.env` поставь `GIGACHAT_VERIFY_SSL=false` (учебный режим OK; в продакшене ставь Russian Trusted Root CA) |
| `401 Unauthorized` | Неверный `GIGACHAT_AUTH_KEY` или scope. Перепроверь на developers.sber.ru и убедись, что ключ скопирован полностью |
| `Generation failed after 3 attempts` | Только при `dm synth generate` (legacy). UI и `dm extract run` этого не делают |
| Кнопки в UI не красятся | Hard-refresh: `Cmd+Shift+R` / `Ctrl+Shift+R`. Streamlit кеширует JS-bundle |
| Wizard повторно открывается | Нажми «Готово» на последнем шаге. X в углу не запоминает закрытие |
| `dm` команда не найдена | Запускай через `uv run dm ...` (не голый `dm`) |

## Дополнительно

- **Архитектура**: `CLAUDE.md` в корне + `docs/roadmap.md` (что будет после MVP-1)
- **Концепция продукта**: Gate 2 PDF (не в репо — у владельца)
- **Issues / PR**: [github.com/Goshamorq/deal_memory](https://github.com/Goshamorq/deal_memory)
