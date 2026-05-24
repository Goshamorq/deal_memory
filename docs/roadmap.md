# DealMemory roadmap

## MVP-1 (текущая итерация)
Thin slice: synth → extract → eval → UI на синтетических русскоязычных диалогах.
План: см. `/Users/goshamorq/.claude/plans/mighty-baking-spring.md` или Gate 2 PDF.

## После MVP-1

- **MVP-2 — STT-слой.** Тест 3 провайдеров (SaluteSpeech / YandexSpeechKit / T-One) на reproducible audio. Закрывает D4 риск #1 (WER на IT-терминах).
- **MVP-3 — Bitrix24 integration.** Webhook на новый звонок → REST API для обновления карточки сделки. Виджет в карточке (макет 1 из Gate 2 C2).
- **MVP-4 — Risk scoring + дашборд РОПа.** Rule-based: дней без контакта, неотработанные возражения, просроченные обещания, частота «дорого». Дашборд воронки (макет 3).
- **MVP-5 — Human-in-the-loop follow-up.** Email-черновик с подтверждением менеджера. Учёт ФЗ-38 (не отправляем без согласия).

## Backlog (после первого пилота)
- amoCRM (после Bitrix24)
- Видеовстречи (Zoom/Teams)
- Multi-tenant
- Биллинг / подписка
