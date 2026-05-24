"""Streamlit UI — 3 tabs: Диалоги / Метрики / Настройки.

Layout per tab is intentionally flat — no helper components, no sidebar
navigation. State lives in session_state for the current selection;
predictions and annotations persist as JSONL on disk.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from deal_memory import annotations as ann_mod
from deal_memory import extract, metrics, storage
from deal_memory.gigachat import GigaChatClient
from deal_memory.schema import (
    FIELD_KEYS,
    Annotation,
    DealFacts,
    Dialog,
    FieldEvidence,
    Prediction,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
POOLS_DIR = REPO_ROOT / "data" / "pools"
EVAL_DIR = REPO_ROOT / "data" / "eval-runs"
ANN_DIR = REPO_ROOT / "data" / "annotations"
CONFIG_DIR = REPO_ROOT / "data" / "config"
PROMPT_FILE = CONFIG_DIR / "prompt.txt"

load_dotenv(REPO_ROOT / ".env")

FIELD_LABELS = {
    "budget": "Бюджет",
    "decision_maker": "ЛПР",
    "technical_requirements": "Тех. требования",
    "objections": "Возражения",
    "promises": "Обещания",
    "next_step": "Следующий шаг",
}
LIST_FIELDS = {"objections", "promises"}
VERDICT_LABEL = {"correct": "✓", "partial": "±", "wrong": "✗", None: "·"}
VERDICT_COLOR = {"correct": "#3a8a3a", "partial": "#c89c1f", "wrong": "#b03a3a"}


# ---- Speaker-aware transcript renderer ----

_SPEAKER_RE = re.compile(
    r"(Менеджер|Клиент|Технический директор клиента|Новый менеджер|ФД клиента|Финансовый директор)\s*:",
    re.IGNORECASE,
)


def split_transcript_turns(transcript: str) -> list[tuple[str, str]]:
    """Return [(speaker, text), ...] regardless of whether the source uses
    newlines or single-paragraph inline-speakers formatting."""
    turns: list[tuple[str, str]] = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            sp, _, rest = line.partition(":")
            turns.append((sp.strip(), rest.strip()))
        elif turns:
            turns[-1] = (turns[-1][0], turns[-1][1] + " " + line)
        else:
            turns.append(("?", line))
    if len(turns) <= 1 and _SPEAKER_RE.search(transcript):
        parts = _SPEAKER_RE.split(transcript)
        turns = []
        i = 1
        while i < len(parts) - 1:
            sp = parts[i].strip()
            body = parts[i + 1].strip()
            if body:
                turns.append((sp, body))
            i += 2
    return turns


def render_chat(transcript: str) -> None:
    for speaker, text in split_transcript_turns(transcript):
        role = "user" if speaker.lower().startswith("клиент") else "assistant"
        with st.chat_message(role):
            st.markdown(f"**{speaker}**  \n{text}")


# ---- Data layer ----


def _list_pools() -> list[str]:
    if not POOLS_DIR.exists():
        return []
    return sorted(p.stem for p in POOLS_DIR.glob("*.jsonl"))


def _load_dialogs(pool: str) -> list[Dialog]:
    return storage.load_jsonl(POOLS_DIR / f"{pool}.jsonl", Dialog)


def _predictions_path(pool: str) -> Path:
    return EVAL_DIR / f"{pool}.jsonl"


def _load_predictions(pool: str) -> dict[str, Prediction]:
    path = _predictions_path(pool)
    if not path.exists():
        return {}
    return {p.id: p for p in storage.load_jsonl(path, Prediction)}


def _save_prediction(pool: str, pred: Prediction) -> None:
    current = _load_predictions(pool)
    current[pred.id] = pred
    storage.write_jsonl(_predictions_path(pool), current.values())


def _load_annotations(pool: str) -> dict[tuple[str, str], Annotation]:
    return ann_mod.load_pool(pool, base=ANN_DIR)


# ---- Field display ----


def _field_text(value: str | None, quote: str | None) -> str:
    if value is None:
        return "_не выявлено_"
    quote_block = f"  \n_«{quote}»_" if quote else ""
    return f"{value}{quote_block}"


def _list_text(items: list[FieldEvidence]) -> str:
    if not items:
        return "_не выявлено_"
    return "\n\n".join(f"- {_field_text(it.value, it.quote)}" for it in items)


def _render_field(
    pool: str,
    dialog_id: str,
    field: str,
    prediction: DealFacts | None,
    current_verdict: str | None,
) -> None:
    label = FIELD_LABELS[field]
    color = VERDICT_COLOR.get(current_verdict, "#888888") if current_verdict else "#888888"

    with st.container(border=True):
        st.markdown(f"**{label}**")
        if prediction is None:
            st.markdown("_(не обработано)_")
        elif field in LIST_FIELDS:
            st.markdown(_list_text(getattr(prediction, field)))
        else:
            fe: FieldEvidence = getattr(prediction, field)
            st.markdown(_field_text(fe.value, fe.quote))

        cols = st.columns([1, 1, 1, 4])
        with cols[0]:
            if st.button(
                "✗",
                key=f"v-wrong-{dialog_id}-{field}",
                type="primary" if current_verdict == "wrong" else "secondary",
                help="Неверно",
                use_container_width=True,
                disabled=prediction is None,
            ):
                ann_mod.upsert(pool, dialog_id, field, "wrong", base=ANN_DIR)
                st.rerun()
        with cols[1]:
            if st.button(
                "±",
                key=f"v-partial-{dialog_id}-{field}",
                type="primary" if current_verdict == "partial" else "secondary",
                help="Частично",
                use_container_width=True,
                disabled=prediction is None,
            ):
                ann_mod.upsert(pool, dialog_id, field, "partial", base=ANN_DIR)
                st.rerun()
        with cols[2]:
            if st.button(
                "✓",
                key=f"v-correct-{dialog_id}-{field}",
                type="primary" if current_verdict == "correct" else "secondary",
                help="Верно",
                use_container_width=True,
                disabled=prediction is None,
            ):
                ann_mod.upsert(pool, dialog_id, field, "correct", base=ANN_DIR)
                st.rerun()
        with cols[3]:
            verdict_str = VERDICT_LABEL.get(current_verdict, "·")
            st.markdown(
                f"<div style='padding:6px 0;color:{color};font-weight:600'>"
                f"Разметка: {verdict_str}</div>",
                unsafe_allow_html=True,
            )


# ---- Tab 1: Диалоги ----


def tab_dialogs() -> None:
    pools = _list_pools()
    if not pools:
        st.warning("Нет пулов в data/pools/. Добавь хотя бы один JSONL-файл.")
        return

    col_pool, col_dialog = st.columns(2)
    with col_pool:
        pool = st.selectbox("Пул диалогов", pools, key="pool_select")
    dialogs = _load_dialogs(pool)
    predictions = _load_predictions(pool)
    annotations = _load_annotations(pool)

    with col_dialog:
        dialog_options = [d.id for d in dialogs]

        def fmt(did: str) -> str:
            d = next(x for x in dialogs if x.id == did)
            mark = "✓" if did in predictions else "·"
            scn = f"  ({d.scenario})" if d.scenario else ""
            return f"{mark}  {did}{scn}"

        dialog_id = st.selectbox("Диалог", dialog_options, format_func=fmt, key="dialog_select")
    if dialog_id is None:
        return
    dialog = next(d for d in dialogs if d.id == dialog_id)

    left, right = st.columns([3, 2])
    with left:
        st.caption(f"Сценарий: **{dialog.scenario or '—'}**")
        render_chat(dialog.transcript)
    with right:
        pred = predictions.get(dialog_id)

        action_col1, action_col2 = st.columns([1, 1])
        with action_col1:
            label = "Переобработать" if pred is not None else "Обработать"
            if st.button(label, type="primary", use_container_width=True):
                try:
                    with st.spinner("GigaChat: extracting..."):
                        with GigaChatClient() as client:
                            new_pred = extract.extract_one(client, dialog.id, dialog.transcript)
                        _save_prediction(pool, new_pred)
                except Exception as exc:
                    st.error(f"Extraction failed: {exc}")
                else:
                    st.rerun()
        with action_col2:
            if pred is not None and st.button("Очистить предсказание", use_container_width=True):
                cur = _load_predictions(pool)
                cur.pop(dialog_id, None)
                if cur:
                    storage.write_jsonl(_predictions_path(pool), cur.values())
                else:
                    _predictions_path(pool).unlink(missing_ok=True)
                st.rerun()

        if pred is not None and pred.parse_repaired:
            st.info("Это предсказание прошло через repair-pass.")

        for f in FIELD_KEYS:
            current = annotations.get((dialog_id, f))
            _render_field(
                pool,
                dialog_id,
                f,
                pred.prediction if pred is not None else None,
                current.verdict if current is not None else None,
            )


# ---- Tab 2: Метрики ----


def tab_metrics() -> None:
    pools = _list_pools()
    if not pools:
        st.warning("Нет пулов.")
        return
    pool = st.selectbox("Пул", pools, key="metrics_pool")
    annotations = _load_annotations(pool)
    if not annotations:
        st.info(
            "Для этого пула пока нет аннотаций. Перейди на вкладку «Диалоги», "
            "обработай диалог и проставь ✓/±/✗ по полям."
        )
        return

    report = metrics.compute_pool_report(pool, annotations)

    c1, c2, c3 = st.columns(3)
    c1.metric("Всего разметок", report.n_annotations)
    c2.metric("Macro accuracy", f"{report.macro_accuracy:.1%}")
    c3.metric("Soft accuracy (± = 0.5)", f"{report.macro_soft_accuracy:.1%}")

    df = pd.DataFrame(
        [
            {
                "поле": FIELD_LABELS[fs.field],
                "✓": fs.correct,
                "±": fs.partial,
                "✗": fs.wrong,
                "всего": fs.total,
                "accuracy": fs.accuracy,
                "soft": fs.soft_accuracy,
                "таргет": fs.target,
                "🎯": "🟢" if fs.hits_target else ("⚪" if fs.total == 0 else "🔴"),
            }
            for fs in report.fields
        ]
    )
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Распределение по полям")
    chart_df = pd.DataFrame(
        {
            FIELD_LABELS[fs.field]: [fs.correct, fs.partial, fs.wrong]
            for fs in report.fields
        },
        index=["✓", "±", "✗"],
    ).T
    st.bar_chart(chart_df)


# ---- Tab 3: Настройки ----


def tab_settings() -> None:
    st.subheader("System prompt")
    st.caption(
        f"Хранится в `{PROMPT_FILE.relative_to(REPO_ROOT)}`. "
        "Применяется при каждом нажатии «Обработать» — без рестартов."
    )
    current_text = extract.load_system_prompt(PROMPT_FILE)
    new_text = st.text_area(
        "Содержимое",
        value=current_text,
        height=400,
        key="prompt_textarea",
    )
    col_save, col_reset, _ = st.columns([1, 1, 4])
    with col_save:
        if st.button("Сохранить", type="primary", use_container_width=True):
            extract.save_system_prompt(new_text, PROMPT_FILE)
            st.success("Сохранено.")
    with col_reset:
        if st.button("Восстановить дефолт", use_container_width=True):
            extract.save_system_prompt(extract.DEFAULT_SYSTEM_PROMPT, PROMPT_FILE)
            st.rerun()

    st.divider()
    st.subheader("Окружение GigaChat")
    env_rows = [
        ("GIGACHAT_MODEL", os.environ.get("GIGACHAT_MODEL", "GigaChat")),
        ("GIGACHAT_SCOPE", os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")),
        ("GIGACHAT_VERIFY_SSL", os.environ.get("GIGACHAT_VERIFY_SSL", "false")),
        ("GIGACHAT_AUTH_KEY", "(set)" if os.environ.get("GIGACHAT_AUTH_KEY") else "(MISSING)"),
    ]
    st.dataframe(
        pd.DataFrame(env_rows, columns=["переменная", "значение"]),
        hide_index=True,
        use_container_width=True,
    )

    st.divider()
    st.subheader("Поля и пороги")
    st.dataframe(
        pd.DataFrame(
            [(FIELD_LABELS[f], metrics.FIELD_TARGETS[f]) for f in FIELD_KEYS],
            columns=["поле", "таргет accuracy"],
        ),
        hide_index=True,
        use_container_width=True,
    )


# ---- Main ----


def main() -> None:
    st.set_page_config(page_title="DealMemory", layout="wide")
    st.title("DealMemory")
    tab1, tab2, tab3 = st.tabs(["Диалоги", "Метрики", "Настройки"])
    with tab1:
        tab_dialogs()
    with tab2:
        tab_metrics()
    with tab3:
        tab_settings()


main()
