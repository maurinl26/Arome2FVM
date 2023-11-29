# -*- coding: utf-8 -*-
import typer

from pathlib import Path
import sys

sys.path.append(str(Path(Path.cwd(), "src")))
sys.path.append(str(Path(Path.cwd().parent.absolute(), "FVM_GT4Py_slim", "src")))
print(sys.path)

from arome2fvm.arome import Arome
from arome2fvm.writer import write_state


def main(arome_file: str, data_file: str):
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
    arome2fvm = Arome(arome_file)

    if data_file is not None:
        write_state(arome2fvm, data_file)


if __name__ == "__main__":
    typer.run(main)
