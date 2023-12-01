# -*- coding: utf-8 -*-
import numpy as np
from plot.plot import plot_zcr
import typer
from typing import Annotated
import netCDF4 as nc
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(Path.cwd(), "src")))
sys.path.append(str(Path(Path.cwd().parent.absolute(), "FVM_GT4Py_slim", "src")))

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = typer.Typer()


@app.command()
def plot_z_coordinate(
    data_file: Annotated[str, typer.Option(help="file name of post-processed data")],
    fig_file: Annotated[str, typer.Option(help="file name to save image")],
):
    ds = nc.Dataset(data_file)
    zcr = ds["zcr"][...]
    xc = ds["x"][...]

    logging.info(
        f"Z coordinate - min, max : {np.min(zcr[:, :, 0])}, {np.min(zcr[:, :, zcr.shape[2] - 1])}"
    )
    logging.info

    fig, _ = plot_zcr(zcr, xc, zcr.shape[2])
    fig.savefig(fig_file)


if __name__ == "__main__":
    app()
