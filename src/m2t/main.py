import typer
import importlib.metadata
from typing import Annotated
from pathlib import Path
from PIL import Image
import hyperspy.api as hs
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn, TimeRemainingColumn
import matplotlib.pyplot as plt
import numpy as np

app = typer.Typer()
version = importlib.metadata.version("m2t")


def version_callback(value: bool):
    if value:
        print(f"M2T CLI Version: {version}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            help="Print the current version.",
            allow_dash=True,
        ),
    ] = None,
):
    pass


@app.command(name="version")
def _version():
    """Print the current version."""
    version_callback(True)

def robust_uint8_conversion(data):
    # 1. Find the 2nd and 98th percentile (ignores outliers)
    vmin, vmax = np.percentile(data, (2, 98))
    
    # 2. Clip the data to these bounds
    data_clipped = np.clip(data, vmin, vmax)
    
    # 3. Normalize to 0-1 range
    # Avoid divide by zero if image is flat
    if vmax != vmin:
        norm = (data_clipped - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(data_clipped)
        
    # 4. Scale to 255 and cast
    return (norm * 255).astype(np.uint8)


@app.command()
def convert(files: list[Path]):
     with Progress(
        SpinnerColumn(finished_text="[green]\u2713[/green]"),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(elapsed_when_finished=True),
    ) as progress:
        converting = progress.add_task(description="Converting", total=len(files))
        for file in files:
            s = hs.load(file, lazy=True)
            s.map(robust_uint8_conversion, inplace=True)
            s.change_dtype("uint8")
            s.save(f"{file.stem}.tif", overwrite=True)
            progress.update(converting, advance=1)

if __name__ == "__main__":
    app()
