"""Typer entry point. Subcommands filled in per phase."""
from __future__ import annotations

import typer

app = typer.Typer(help="DealMemory MVP-1 — synth/extract/eval/ui pipeline.")

synth_app = typer.Typer(help="Synthetic dialog generation.")
extract_app = typer.Typer(help="LLM extraction over transcripts.")
eval_app = typer.Typer(help="Score predictions vs ground truth.")
app.add_typer(synth_app, name="synth")
app.add_typer(extract_app, name="extract")
app.add_typer(eval_app, name="eval")


@app.command()
def ui(port: int = 8501) -> None:
    """Launch Streamlit UI (Phase E)."""
    import subprocess
    from pathlib import Path

    ui_path = Path(__file__).with_name("ui.py")
    subprocess.run(["streamlit", "run", str(ui_path), "--server.port", str(port)], check=False)


if __name__ == "__main__":
    app()
