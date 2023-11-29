# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

sys.path.append(str(Path(Path.cwd(), "src")))
sys.path.append(str(Path(Path.cwd().parent.absolute(), "FVM_GT4Py_slim", "src")))

from arome import Arome


if __name__ == "__main__":

    arome_file = os.path.join(os.getcwd(), "files", "nc_files", "historic.arome.nc")

    config_file = os.path.join(os.getcwd(), "config", "alps.yml")

    arome = Arome(arome_file, config_file)

    # xrc = arome.grid.xcr
