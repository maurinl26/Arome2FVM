# -*- coding: utf-8 -*-
from typing import Annotated
import typer
import logging
import netCDF4 as nc
from pathlib import Path
import numpy as np

from arome2fvm.arome2fvm import Arome2FVM
from arome2fvm.writer import write_state
from utils.plots import plot_zcr

app = typer.Typer()

@app.command()
def convert_vertical_coordinate(
    arome_file: Annotated[Path, typer.Option(help=".nc datafile with Arome2FVM raw fields")],
    data_file: Annotated[
        Path, typer.Option(help=".nc datafile to dump post-processed file")
    ],
):
    """Read .yml configuration
    1. Extract .nc file (issued from AROME .fa state)
    2. Perform conversions
    3. Write state in a .nc file which can be read by FVM


    Args:
        arome_file (str): arome raw data (.nc)
        config_file (str): config file (.yml) from FVM
        data_file (str): output file containing vertical coordinates and orography
    """

    # Reads AROME file
    arome2fvm = Arome2FVM(arome_file)

    if data_file is not None:
        write_state(arome2fvm, data_file)
        
        
@app.command()
def plot_vertical_coordinate(
    data_file: Annotated[Path, typer.Option(help="file name of post-processed data")],
    fig_file: Annotated[Path, typer.Option(help="file name to save image")],
):
    ds = nc.Dataset(data_file)
    zcr = ds["zcr"][...]
    xc = ds["x"][...]

    logging.info(
        f"Z coordinate - min, max : {np.min(zcr[:, :, 0])}, {np.min(zcr[:, :, zcr.shape[2] - 1])}"
    )

    fig, _ = plot_zcr(zcr, xc, zcr.shape[2])
    fig.savefig(fig_file)


if __name__ == "__main__":
    app()
