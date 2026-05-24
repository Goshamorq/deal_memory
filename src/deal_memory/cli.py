"""Typer entry point — all subcommands wired up."""
from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from deal_memory import eval as eval_mod
from deal_memory import storage, synth
from deal_memory.gigachat import GigaChatClient
from deal_memory.schema import Prediction, SyntheticSample

load_dotenv()
console = Console()

app = typer.Typer(help="DealMemory MVP-1 — synth/extract/eval/ui pipeline.")

synth_app = typer.Typer(help="Synthetic dialog generation.")
extract_app = typer.Typer(help="LLM extraction over transcripts.")
eval_app = typer.Typer(help="Score predictions vs ground truth.")
app.add_typer(synth_app, name="synth")
app.add_typer(extract_app, name="extract")
app.add_typer(eval_app, name="eval")


@synth_app.command("generate")
def synth_generate(
    n: int = typer.Option(50, "--n", help="Number of dialogs to generate."),
    out: Path = typer.Option(
        Path("data/synthetic/v1.jsonl"), "--out", help="Output JSONL path."
    ),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for reproducibility."),
) -> None:
    """Generate N synthetic B2B IT sales dialogs with ground-truth labels."""
    client = GigaChatClient()
    samples = []
    with console.status(f"Generating {n} dialogs via GigaChat...") as status:
        try:
            for i, sample in enumerate(synth.generate(client, n, seed=seed), start=1):
                samples.append(sample)
                status.update(f"Generated {i}/{n}")
        finally:
            client.close()
    count = storage.write_jsonl(out, samples)
    console.print(f"[green]Wrote {count} samples to {out}[/green]")


def _render_report(report: eval_mod.EvalReport) -> Table:
    t = Table(title=f"Eval report (n={report.n_samples})")
    t.add_column("field")
    t.add_column("TP", justify="right")
    t.add_column("FP", justify="right")
    t.add_column("FN", justify="right")
    t.add_column("precision", justify="right")
    t.add_column("recall", justify="right")
    t.add_column("F1", justify="right", style="bold")
    for fm in report.fields:
        t.add_row(
            fm.field,
            str(fm.tp),
            str(fm.fp),
            str(fm.fn),
            f"{fm.precision:.2f}",
            f"{fm.recall:.2f}",
            f"{fm.f1:.2f}",
        )
    return t


@eval_app.command("score")
def eval_score(
    truth_path: Path = typer.Option(
        ..., "--truth", help="Synthetic-samples JSONL with ground truth."
    ),
    predictions: Path = typer.Option(
        ..., "--predictions", help="Predictions JSONL from `dm extract`."
    ),
    report_out: Path | None = typer.Option(
        None, "--report-out", help="Optional path to dump the report as JSON."
    ),
) -> None:
    """Score predictions vs ground truth; prints per-field F1 + hallucination rate."""
    truth = storage.load_jsonl(truth_path, SyntheticSample)
    preds = storage.load_jsonl(predictions, Prediction)
    report = eval_mod.score(truth, preds)

    console.print(_render_report(report))
    console.print(
        f"\n[bold]Macro F1:[/bold] {report.macro_f1:.3f}    "
        f"[bold]Hallucination rate:[/bold] {report.hallucination_rate:.1%} "
        f"({report.hallucination_count}/{report.hallucination_total})    "
        f"[bold]Coverage:[/bold] {report.coverage:.1%} "
        f"({report.coverage_correct}/{report.coverage_total})"
    )
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Report written to {report_out}[/green]")


@app.command()
def ui(port: int = 8501) -> None:
    """Launch Streamlit UI."""
    import subprocess

    ui_path = Path(__file__).with_name("ui.py")
    subprocess.run(["streamlit", "run", str(ui_path), "--server.port", str(port)], check=False)


if __name__ == "__main__":
    app()
