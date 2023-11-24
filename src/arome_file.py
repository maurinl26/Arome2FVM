# -*- coding: utf-8 -*-
from dataclasses import dataclass
from pathlib import Path
import netCDF4 as nc
import numpy as np

from levels import Levels


class AromeReader:

    ds: nc.Dataset
    level_order: Levels = Levels.TOP_TO_BOTTOM

    def __init__(self, filename: str):
        self.ds = nc.Dataset(filename)

    def get_nx(self):
        return self.ds.dimensions["X"].size

    def get_ny(self):
        return self.ds.dimensions["Y"].size

    def get_nz(self):
        return self.ds.dimensions["Z"].size

    def get_n_interface_levels(self):
        return self.ds.dimensions["Z+1"].size
