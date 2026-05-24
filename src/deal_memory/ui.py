"""Streamlit UI — three pages on top of synth + extract + eval data.

Run via `dm ui` (which spawns `streamlit run` on this file).

No write paths, no auth, no editing — read-only inspector of local JSONL
artefacts produced by the CLI commands.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from deal_memory import eval as eval_mod
from deal_memory import storage
from deal_memory.schema import DealFacts, FieldEvidence, Prediction, SyntheticSample

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
EVAL_DIR = REPO_ROOT / "data" / "eval-runs"
SAMPLE_DIR = REPO_ROOT / "data" / "sample"

FIELD_TARGETS = {
    "budget": 0.85,
    "decision_maker": 0.75,
    "technical_requirements": 0.70,
    "objections": 0.65,
    "promises": 0.75,
    "next_step": 0.80,
}


# ---- Data helpers ----


@st.cache_data
def list_synthetic_files() -> list[Path]:
    files: list[Path] = []
    for d in (SYNTHETIC_DIR, SAMPLE_DIR):
        if d.exists():
            files.extend(sorted(d.glob("*.jsonl")))
    return files


@st.cache_data
def list_eval_files() -> list[Path]:
    runs = storage.list_eval_runs(EVAL_DIR)
    if SAMPLE_DIR.exists():
        runs.extend(sorted(SAMPLE_DIR.glob("predictions*.jsonl")))
    return runs


@st.cache_data
def load_samples(path_str: str) -> list[SyntheticSample]:
    return storage.load_jsonl(Path(path_str), SyntheticSample)


@st.cache_data
def load_predictions(path_str: str) -> list[Prediction]:
    return storage.load_jsonl(Path(path_str), Prediction)


# ---- Rendering helpers ----


def _format_field(fe: FieldEvidence) -> str:
    if fe.value is None:
        return "_не упомянуто_"
    quote_line = f"  \n_«{fe.quote}»_" if fe.quote else ""
    return f"**{fe.value}**{quote_line}"


def _format_list(items: list[FieldEvidence]) -> str:
    if not items:
        return "_не упомянуто_"
    return "\n\n".join(f"- {_format_field(it)}" for it in items)


def render_facts(facts: DealFacts) -> None:
    st.markdown(f"**Бюджет:** {_format_field(facts.budget)}")
    st.markdown(f"**ЛПР:** {_format_field(facts.decision_maker)}")
    st.markdown(f"**Тех. требования:** {_format_field(facts.technical_requirements)}")
    st.markdown(f"**Возражения:**\n{_format_list(facts.objections)}")
    st.markdown(f"**Обещания:**\n{_format_list(facts.promises)}")
    st.markdown(f"**Следующий шаг:** {_format_field(facts.next_step)}")


def render_transcript(transcript: str) -> None:
    # Add bold to speaker labels at the start of each line.
    lines = transcript.splitlines()
    md_lines = []
    for line in lines:
        if ":" in line:
            speaker, _, rest = line.partition(":")
            md_lines.append(f"**{speaker.strip()}:**{rest}")
        else:
            md_lines.append(line)
    st.markdown("\n\n".join(md_lines))


# ---- Page: dialogs browser ----


def page_dialogs() -> None:
    st.header("Dialogs browser")
    files = list_synthetic_files()
    if not files:
        st.info("Нет JSONL-файлов в data/synthetic/ или data/sample/. Запусти `dm synth generate`.")
        return

    selected = st.sidebar.selectbox(
        "JSONL файл", files, format_func=lambda p: f"{p.parent.name}/{p.name}"
    )
    samples = load_samples(str(selected))

    scenarios = sorted({s.scenario for s in samples})
    scenario_filter = st.sidebar.multiselect("Сценарий", scenarios, default=scenarios)
    filtered = [s for s in samples if s.scenario in scenario_filter]

    if not filtered:
        st.warning("Нет диалогов под выбранный фильтр.")
        return

    ids = [s.id for s in filtered]
    chosen_id = st.sidebar.selectbox(
        "Диалог", ids, format_func=lambda x: f"{x[:8]}  ({_scenario_for(filtered, x)})"
    )
    sample = next(s for s in filtered if s.id == chosen_id)

    st.caption(f"Сценарий: **{sample.scenario}**   •   persona: `{sample.meta.get('persona_id', '?')}`")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Транскрипт")
        render_transcript(sample.transcript)
    with col_right:
        st.subheader("Ground truth")
        render_facts(sample.ground_truth)


def _scenario_for(samples: list[SyntheticSample], sid: str) -> str:
    return next(s.scenario for s in samples if s.id == sid)


# ---- Page: extraction comparison ----


def _compare_scalar(truth: FieldEvidence, pred: FieldEvidence) -> tuple[str, str]:
    if truth.value is None and pred.value is None:
        return "🟢", "match (оба null)"
    if truth.value is None and pred.value is not None:
        return "🟣", "hallucination"
    if truth.value is not None and pred.value is None:
        return "🔴", "missed"
    if eval_mod.jaccard(truth.value or "", pred.value or "") >= eval_mod.JACCARD_THRESHOLD:
        return "🟢", "match"
    return "🔴", "mismatch"


def _compare_list(truth: list[FieldEvidence], pred: list[FieldEvidence]) -> tuple[str, str]:
    if not truth and not pred:
        return "🟢", "оба пустые"
    if not truth and pred:
        return "🟣", "hallucination (truth пуст)"
    if truth and not pred:
        return "🔴", f"missed {len(truth)} items"
    tp, fp, fn = eval_mod._greedy_match_lists(truth, pred, eval_mod.JACCARD_THRESHOLD)
    if fp == 0 and fn == 0:
        return "🟢", f"match {tp}/{tp}"
    if tp > 0:
        return "🟡", f"partial: TP={tp} FP={fp} FN={fn}"
    return "🔴", f"no overlap: FP={fp} FN={fn}"


def page_comparison() -> None:
    st.header("Extraction comparison")
    eval_files = list_eval_files()
    synth_files = list_synthetic_files()

    if not eval_files:
        st.info("Нет eval-runs в data/eval-runs/. Запусти `dm extract run`.")
        return
    if not synth_files:
        st.info("Нет synthetic-файлов с ground truth.")
        return

    truth_path = st.sidebar.selectbox(
        "Ground-truth файл", synth_files, format_func=lambda p: f"{p.parent.name}/{p.name}"
    )
    pred_path = st.sidebar.selectbox(
        "Eval run", eval_files, format_func=lambda p: p.name
    )
    samples = load_samples(str(truth_path))
    preds = load_predictions(str(pred_path))
    pred_by_id = {p.id: p for p in preds}

    common_ids = [s.id for s in samples if s.id in pred_by_id]
    if not common_ids:
        st.warning("Нет общих id между truth и predictions.")
        return

    chosen = st.sidebar.selectbox("Диалог", common_ids, format_func=lambda x: x[:8])
    sample = next(s for s in samples if s.id == chosen)
    pred = pred_by_id[chosen]

    if pred.parse_repaired:
        st.warning("Этот prediction прошёл через repair-pass.")

    st.caption(f"Сценарий: **{sample.scenario}**")

    # Side-by-side table
    rows = []
    for f in ("budget", "decision_maker", "technical_requirements", "next_step"):
        t_fe: FieldEvidence = getattr(sample.ground_truth, f)
        p_fe: FieldEvidence = getattr(pred.prediction, f)
        emoji, note = _compare_scalar(t_fe, p_fe)
        rows.append(
            {
                "поле": f,
                "статус": emoji,
                "ground truth": t_fe.value or "—",
                "prediction": p_fe.value or "—",
                "комментарий": note,
            }
        )
    for f in ("objections", "promises"):
        t_items = getattr(sample.ground_truth, f)
        p_items = getattr(pred.prediction, f)
        emoji, note = _compare_list(t_items, p_items)
        rows.append(
            {
                "поле": f,
                "статус": emoji,
                "ground truth": "\n".join(it.value or "" for it in t_items) or "—",
                "prediction": "\n".join(it.value or "" for it in p_items) or "—",
                "комментарий": note,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Транскрипт"):
        render_transcript(sample.transcript)
    with st.expander("Raw JSON: ground truth"):
        st.code(sample.ground_truth.model_dump_json(indent=2), language="json")
    with st.expander("Raw JSON: prediction"):
        st.code(pred.prediction.model_dump_json(indent=2), language="json")


# ---- Page: metrics dashboard ----


def page_dashboard() -> None:
    st.header("Metrics dashboard")
    eval_files = list_eval_files()
    synth_files = list_synthetic_files()

    if not eval_files:
        st.info("Нет eval-runs.")
        return

    truth_path = st.sidebar.selectbox(
        "Ground-truth файл", synth_files, format_func=lambda p: f"{p.parent.name}/{p.name}"
    )
    pred_path = st.sidebar.selectbox(
        "Eval run", eval_files, format_func=lambda p: p.name
    )
    samples = load_samples(str(truth_path))
    preds = load_predictions(str(pred_path))

    try:
        report = eval_mod.score(samples, preds)
    except ValueError as exc:
        st.error(str(exc))
        return

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Диалогов", report.n_samples)
    c2.metric("Macro F1", f"{report.macro_f1:.3f}")
    c3.metric(
        "Hallucination",
        f"{report.hallucination_rate:.1%}",
        delta=f"{report.hallucination_count}/{report.hallucination_total}",
        delta_color="inverse",
    )
    c4.metric(
        "Coverage",
        f"{report.coverage:.1%}",
        delta=f"{report.coverage_correct}/{report.coverage_total}",
    )

    # F1 per field
    st.subheader("F1 по полям")
    df = pd.DataFrame(
        [
            {
                "поле": fm.field,
                "TP": fm.tp,
                "FP": fm.fp,
                "FN": fm.fn,
                "precision": fm.precision,
                "recall": fm.recall,
                "F1": fm.f1,
                "таргет": FIELD_TARGETS.get(fm.field, 0.0),
                "🎯": "🟢" if fm.f1 >= FIELD_TARGETS.get(fm.field, 0.0) else "🔴",
            }
            for fm in report.fields
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Hallucinations per field (bar)
    st.subheader("Распределение FP по полям")
    fp_df = pd.DataFrame({fm.field: [fm.fp] for fm in report.fields}).T.rename(columns={0: "FP"})
    st.bar_chart(fp_df)

    # Historical macro F1
    if len(eval_files) > 1:
        st.subheader("История macro F1 по eval-runs")
        history = []
        for f in reversed(eval_files):  # oldest → newest
            try:
                p = load_predictions(str(f))
                r = eval_mod.score(samples, p)
                history.append({"run": f.name, "macro_f1": r.macro_f1})
            except ValueError:
                continue
        if history:
            hdf = pd.DataFrame(history).set_index("run")
            st.line_chart(hdf)


# ---- Main ----


def main() -> None:
    st.set_page_config(page_title="DealMemory", layout="wide")
    st.sidebar.title("DealMemory")
    page = st.sidebar.radio(
        "Страница",
        ["Dialogs browser", "Extraction comparison", "Metrics dashboard"],
    )
    st.sidebar.divider()
    if page == "Dialogs browser":
        page_dialogs()
    elif page == "Extraction comparison":
        page_comparison()
    else:
        page_dashboard()


main()
