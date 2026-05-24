"""Typer entry point. Subcommands filled in per phase."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from deal_memory import extract, storage
from deal_memory.gigachat import GigaChatClient
from deal_memory.schema import SyntheticSample

load_dotenv()
console = Console()

app = typer.Typer(help="DealMemory MVP-1 — synth/extract/eval/ui pipeline.")

synth_app = typer.Typer(help="Synthetic dialog generation.")
extract_app = typer.Typer(help="LLM extraction over transcripts.")
eval_app = typer.Typer(help="Score predictions vs ground truth.")
app.add_typer(synth_app, name="synth")
app.add_typer(extract_app, name="extract")
app.add_typer(eval_app, name="eval")


@extract_app.command("run")
def extract_run(
    in_path: Path = typer.Option(..., "--in", help="Synthetic JSONL with transcripts."),
    out: Path | None = typer.Option(
        None, "--out", help="Output JSONL (default: data/eval-runs/<ts>.jsonl)."
    ),
) -> None:
    """Run extraction over a synthetic-samples JSONL."""
    samples = storage.load_jsonl(in_path, SyntheticSample)
    if out is None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = Path("data/eval-runs") / f"{ts}.jsonl"

    client = GigaChatClient()
    predictions = []
    with console.status(f"Extracting {len(samples)} transcripts via GigaChat...") as status:
        try:
            for i, sample in enumerate(samples, start=1):
                predictions.append(extract.extract_one(client, sample.id, sample.transcript))
                status.update(f"Extracted {i}/{len(samples)}")
        finally:
            client.close()
    count = storage.write_jsonl(out, predictions)
    repaired = sum(1 for p in predictions if p.parse_repaired)
    console.print(
        f"[green]Wrote {count} predictions to {out}[/green] (repair-passes: {repaired})"
    )


@app.command()
def ui(port: int = 8501) -> None:
    """Launch Streamlit UI (Phase E)."""
    import subprocess

    ui_path = Path(__file__).with_name("ui.py")
    subprocess.run(["streamlit", "run", str(ui_path), "--server.port", str(port)], check=False)


if __name__ == "__main__":
    app()
