# -*- coding: utf-8 -*-
import typer

from arome2fvm.arome import Arome
from arome2fvm.writer import write_state


def main(arome_file: str, config_file: str, data_file: str):
    """Read .yml configuration
    

    Args:
        arome_file (str): arome raw data (.nc)
        config_file (str): config file (.yml) from FVM
        data_file (str): output file containing vertical coordinates and orography
    """

    # Reads AROME file
    config = Arome(arome_file, config_file)

    if data_file is not None:
        write_state(config, data_file)


if __name__ == "__main__":
    typer.run(main)
