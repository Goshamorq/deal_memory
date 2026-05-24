# Quickstart

## 1. Поставь зависимости

Нужен Python 3.11+, git и `uv`. Если `uv` нет:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Склонируй репо

```bash
git clone https://github.com/Goshamorq/deal_memory.git
cd deal_memory
```

## 3. Установи зависимости

```bash
uv sync --all-extras
```

## 4. Получи GigaChat API key

Открой [developers.sber.ru/portal/products/gigachat-api](https://developers.sber.ru/portal/products/gigachat-api) → авторизуйся → создай проект → скопируй **Authorization key** (это длинная base64-строка).

## 5. Создай `.env` и вставь ключ

```bash
cp .env.example .env
```

Открой `.env` и впиши свой ключ в `GIGACHAT_AUTH_KEY=...`. Остальное оставь по умолчанию.

Либо одной командой (замени `XXX` на свой ключ):

```bash
echo "GIGACHAT_AUTH_KEY=XXX
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_MODEL=GigaChat
GIGACHAT_VERIFY_SSL=false" > .env
```

## 6. Запусти UI

```bash
uv run dm ui
```

Открой [http://localhost:8501](http://localhost:8501).

## 7. Пройди wizard

При первой загрузке поднимется модал из 4 шагов — он объяснит, что есть в каждой вкладке. Жми «Далее →» до «Готово».

Повторно открыть wizard — фиолетовая кнопка **🪄 Wizard** справа сверху.

## 8. Начни работать

На вкладке «Диалоги» выбери диалог → жми **«Обработать»** → размечай поля кнопками **✗ ± ✓**. Метрики смотри во вкладке «Метрики», system prompt редактируй во вкладке «Настройки».
