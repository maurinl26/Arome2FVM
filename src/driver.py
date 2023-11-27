# -*- coding: utf-8 -*-
import typer

from arome import Arome
from fvms.model.config import config_to_file
from writer import write_state


def main(arome_file: str, config_file: str, data_file: str):

    # Reads AROME file
    config = Arome(arome_file)

    if config_file is not None:
        config_to_file(config)

    if data_file is not None:
        write_state(config, data_file)


if __name__ == "__main__":
    typer.run(main)
