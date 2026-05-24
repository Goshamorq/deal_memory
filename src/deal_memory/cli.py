"""Typer entry point."""
from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from deal_memory import extract, storage
from deal_memory.gigachat import GigaChatClient
from deal_memory.schema import Dialog

load_dotenv()
console = Console()

app = typer.Typer(help="DealMemory MVP-1 — pool-based extraction pipeline.")
extract_app = typer.Typer(help="LLM extraction over a pool of dialogs.")
app.add_typer(extract_app, name="extract")


@extract_app.command("run")
def extract_run(
    pool_path: Path = typer.Option(..., "--in", help="Pool JSONL with Dialog rows."),
    out: Path | None = typer.Option(
        None, "--out", help="Output JSONL (default: data/eval-runs/<pool>.jsonl)."
    ),
) -> None:
    """Batch-extract a whole pool. UI does single-dialog extraction on demand;
    this is for seeding predictions or for CLI workflows."""
    dialogs = storage.load_jsonl(pool_path, Dialog)
    if out is None:
        out = Path("data/eval-runs") / pool_path.name

    client = GigaChatClient(model=extract.load_model_name())
    predictions = []
    with console.status(f"Extracting {len(dialogs)} dialogs via GigaChat...") as status:
        try:
            for i, d in enumerate(dialogs, start=1):
                predictions.append(extract.extract_one(client, d.id, d.transcript))
                status.update(f"Extracted {i}/{len(dialogs)}")
        finally:
            client.close()
    count = storage.write_jsonl(out, predictions)
    repaired = sum(1 for p in predictions if p.parse_repaired)
    console.print(
        f"[green]Wrote {count} predictions to {out}[/green] (repair-passes: {repaired})"
    )


@app.command()
def ui(port: int = 8501) -> None:
    """Launch Streamlit UI."""
    import subprocess

    ui_path = Path(__file__).with_name("ui.py")
    subprocess.run(["streamlit", "run", str(ui_path), "--server.port", str(port)], check=False)


if __name__ == "__main__":
    app()
