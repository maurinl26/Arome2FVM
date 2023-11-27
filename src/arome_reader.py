# -*- coding: utf-8 -*-
from typing import Tuple
import netCDF4 as nc
import numpy as np

from levels import LevelOrder


class AromeReader:

    ds: nc.Dataset
    level_order: LevelOrder = LevelOrder.TOP_TO_BOTTOM

    def __init__(self, filename: str):
        self.ds = nc.Dataset(filename)
        self.nx = self.get_nx()
        self.ny = self.get_ny()
        self.nz = self.get_nz()

    def get_nx(self):
        return self.ds.dimensions["X"].size

    def get_ny(self):
        return self.ds.dimensions["Y"].size

    def get_nz(self):
        return self.ds.dimensions["Z"].size

    def get_dims(self):
        return {"nx": self.get_nx(), "ny": self.get_ny(), "nz": self.get_nz()}

    def get_nz_tilde(self):
        return self.ds.dimensions["Z+1"].size

    def get_vertical_divergence(self) -> np.ndarray:
        vertical_divergence = np.zeros((self.nx, self.ny, self.nz))
        for i in range(self.nz):
            vertical_divergence[:, :, i] = self.ds[f"S{self.nz - i:0>3}VERTIC.DIVER"][
                ...
            ].T

        return vertical_divergence

    def get_surface_velocities(self) -> Tuple[np.ndarray]:
        return self.ds["S090WIND.U.PHYS"][...], self.ds["S090WIND.V.PHYS"][...]

    def get_surface_geopotential(self) -> np.ndarray:
        return self.ds["SPECSURFGEOPOTEN"][...]

    def get_temperature(self) -> np.ndarray:
        temperature = np.zeros((self.nx, self.ny, self.nz))

        for i in range(self.nz):
            temperature[:, :, i] = self.ds[f"S{self.nz - i:0>3}TEMPERATURE"][...].T

        return temperature

    def get_pressure(self) -> np.ndarray:
        nh_pressure = np.zeros((self.nx, self.ny, self.nz))
        for i in range(self.nz):
            nh_pressure[:, :, i] = self.ds[f"S{self.nz - i:0>3}PRESS.DEPART"][...].T

        return nh_pressure

    def get_horizontal_velocities(self) -> Tuple[np.ndarray]:
        uvel = np.zeros((self.nx, self.ny, self.nz))
        vvel = np.zeros((self.nx, self.ny, self.nz))

        for i in range(self.nz):
            uvel[:, :, i] = self.ds[f"S{self.nz - i:0>3}WIND.U.PHYS"][...].T
            vvel[:, :, i] = self.ds[f"S{self.nz - i:0>3}WIND.V.PHYS"][...].T

        return uvel, vvel

    def get_hybrid_coef_A(self):
        return self.ds["hybrid_coef_A"][...]

    def get_hybrid_coef_B(self):
        return self.ds["hybrid_coef_B"][...]

    def get_surface_pressure(self):
        return self.ds["SURFPRESSION"][...]
