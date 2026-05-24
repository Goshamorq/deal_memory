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

# Annotation row layout: [title, ✗, ±, ✓] in 4 columns.
# Streamlit's `type="primary"` does NOT add an HTML attribute (the `kind`
# prop is emotion-styled and stays React-only), so we instead emit a
# hidden marker <div class="dm-verdict-X"> BEFORE the columns block. CSS
# uses :has(marker) + adjacent-sibling to color the matching column's
# button. Modern :has() works in Chrome 105+/Safari 15.4+/Firefox 121+.
ANNOTATION_CSS = """
<style>
.dm-verdict { display: none; }

/* Compact + neutral baseline for annotation buttons. Scoped to 4-col blocks
   where the 4th column exists and the 5th doesn't — unique to our row. */
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(4))
:not(:has(> div[data-testid="stColumn"]:nth-child(5)))
> div[data-testid="stColumn"]:nth-child(n+2) button {
    padding: 0.15rem 0.4rem !important;
    min-height: 0 !important;
    line-height: 1 !important;
    font-size: 14px !important;
    border-width: 1px !important;
    background: #ffffff !important;
    color: #555 !important;
    border-color: #d0d0d0 !important;
}

/* Hover tints by column position (2=red ✗, 3=yellow ±, 4=green ✓) */
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(4))
:not(:has(> div[data-testid="stColumn"]:nth-child(5)))
> div[data-testid="stColumn"]:nth-child(2) button:hover {
    background: #fce4e4 !important; color: #b03a3a !important; border-color: #b03a3a !important;
}
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(4))
:not(:has(> div[data-testid="stColumn"]:nth-child(5)))
> div[data-testid="stColumn"]:nth-child(3) button:hover {
    background: #fcefd0 !important; color: #8a6a14 !important; border-color: #c89c1f !important;
}
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(4))
:not(:has(> div[data-testid="stColumn"]:nth-child(5)))
> div[data-testid="stColumn"]:nth-child(4) button:hover {
    background: #dff0d8 !important; color: #2e6b2e !important; border-color: #3a8a3a !important;
}

/* Active verdict: marker <div class="dm-verdict-wrong|partial|correct"> is
   rendered as a sibling element-container immediately BEFORE the columns
   block. CSS then targets the next sibling's matching column button. */
div[data-testid="stElementContainer"]:has(.dm-verdict-wrong) + div[data-testid="stElementContainer"]
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) button {
    background: #b03a3a !important; color: #ffffff !important; border-color: #b03a3a !important;
}
div[data-testid="stElementContainer"]:has(.dm-verdict-partial) + div[data-testid="stElementContainer"]
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) button {
    background: #c89c1f !important; color: #ffffff !important; border-color: #c89c1f !important;
}
div[data-testid="stElementContainer"]:has(.dm-verdict-correct) + div[data-testid="stElementContainer"]
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(4) button {
    background: #3a8a3a !important; color: #ffffff !important; border-color: #3a8a3a !important;
}
</style>
"""


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

    with st.container(border=True):
        # Marker for CSS sibling-targeting: tells the very-next-sibling
        # element-container (the columns block) which annotation column
        # should render as "active" (filled with its colour).
        if current_verdict:
            st.markdown(
                f'<div class="dm-verdict dm-verdict-{current_verdict}"></div>',
                unsafe_allow_html=True,
            )

        # Header row: 4 cols — [title, ✗, ±, ✓].
        title_col, c_wrong, c_partial, c_correct = st.columns([6, 1, 1, 1])
        with title_col:
            st.markdown(f"**{label}**")
        with c_wrong:
            if st.button(
                "✗",
                key=f"v-wrong-{dialog_id}-{field}",
                help="Неверно",
                use_container_width=True,
                disabled=prediction is None,
            ):
                ann_mod.upsert(pool, dialog_id, field, "wrong", base=ANN_DIR)
                st.rerun()
        with c_partial:
            if st.button(
                "±",
                key=f"v-partial-{dialog_id}-{field}",
                help="Частично",
                use_container_width=True,
                disabled=prediction is None,
            ):
                ann_mod.upsert(pool, dialog_id, field, "partial", base=ANN_DIR)
                st.rerun()
        with c_correct:
            if st.button(
                "✓",
                key=f"v-correct-{dialog_id}-{field}",
                help="Верно",
                use_container_width=True,
                disabled=prediction is None,
            ):
                ann_mod.upsert(pool, dialog_id, field, "correct", base=ANN_DIR)
                st.rerun()

        # Body row: prediction content (full width)
        if prediction is None:
            st.markdown("_(не обработано)_")
        elif field in LIST_FIELDS:
            st.markdown(_list_text(getattr(prediction, field)))
        else:
            fe: FieldEvidence = getattr(prediction, field)
            st.markdown(_field_text(fe.value, fe.quote))


# ---- Tab 1: Диалоги ----


def tab_dialogs() -> None:
    pools = _list_pools()
    if not pools:
        st.warning("Нет пулов в data/pools/. Добавь хотя бы один JSONL-файл.")
        return

    col_pool, col_dialog = st.columns(2)
    with col_pool:
        pool_idx = pools.index(st.session_state["pool_select"]) if st.session_state.get("pool_select") in pools else 0
        pool = st.selectbox("Пул диалогов", pools, index=pool_idx, key="pool_select")
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

        # Persist selection across st.rerun() — relying on key= alone proved
        # flaky when buttons inside the right column trigger a rerun.
        saved_did = st.session_state.get("dialog_select")
        dialog_idx = dialog_options.index(saved_did) if saved_did in dialog_options else 0
        dialog_id = st.selectbox(
            "Диалог",
            dialog_options,
            index=dialog_idx,
            format_func=fmt,
            key="dialog_select",
        )
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
                "✗": fs.wrong,
                "±": fs.partial,
                "✓": fs.correct,
                "всего": fs.total,
                "accuracy": fs.accuracy,
                "soft": fs.soft_accuracy,
                "таргет": fs.target,
                "🎯": "🟢" if fs.hits_target else ("⚪" if fs.total == 0 else "🔴"),
            }
            for fs in report.fields
        ]
    )

    styled = df.style.format({"accuracy": "{:.0%}", "soft": "{:.0%}", "таргет": "{:.0%}"})
    st.dataframe(styled, hide_index=True, use_container_width=True)

    st.subheader("Распределение по полям")
    chart_df = pd.DataFrame(
        {
            FIELD_LABELS[fs.field]: [fs.wrong, fs.partial, fs.correct]
            for fs in report.fields
        },
        index=["✗", "±", "✓"],
    ).T
    # Lighter shades of the button colors so chart stays readable but soft.
    st.bar_chart(chart_df, color=["#e08585", "#e0c060", "#85c685"])


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
    st.markdown(ANNOTATION_CSS, unsafe_allow_html=True)
    st.title("DealMemory")
    tab1, tab2, tab3 = st.tabs(["Диалоги", "Метрики", "Настройки"])
    with tab1:
        tab_dialogs()
    with tab2:
        tab_metrics()
    with tab3:
        tab_settings()


main()
