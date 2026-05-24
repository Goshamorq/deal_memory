"""Typer entry point. Subcommands filled in per phase."""
from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from deal_memory import storage, synth
from deal_memory.gigachat import GigaChatClient

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


@app.command()
def ui(port: int = 8501) -> None:
    """Launch Streamlit UI (Phase E)."""
    import subprocess
    from pathlib import Path

    ui_path = Path(__file__).with_name("ui.py")
    subprocess.run(["streamlit", "run", str(ui_path), "--server.port", str(port)], check=False)


if __name__ == "__main__":
    app()
