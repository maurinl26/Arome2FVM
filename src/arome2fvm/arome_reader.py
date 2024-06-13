# -*- coding: utf-8 -*-
from typing import Tuple
import netCDF4 as nc
import numpy as np


class AromeReader:

    ds: nc.Dataset

    def __init__(self, filename: str):
        self.ds = nc.Dataset(filename)
        self.nx = self.get_nx()
        self.ny = self.get_ny()
        self.nz = self.get_nz()
        self.nz_faces = self.get_nz_faces()

    def get_nx(self):
        return self.ds.dimensions["X"].size

    def get_ny(self):
        return self.ds.dimensions["Y"].size

    def get_nz(self):
        return self.ds.dimensions["Z"].size

    def get_nz_faces(self):
        return self.ds.dimensions["Z+1"].size

    def get_spacing(self):
        dx = self.ds["Projection_parameters"].x_resolution
        dy = self.ds["Projection_parameters"].y_resolution

        return dx, dy

    def get_dims(self):
        return {"nx": self.nx, "ny": self.ny, "nz": self.nz}

    def get_dims_interfaces(self):
        return {"nx": self.nx, "ny": self.ny, "nz+1": self.nz_faces}

    def get_vertical_divergence(self) -> np.ndarray:
        vertical_divergence = np.zeros((self.nx, self.ny, self.nz))
        for i in range(self.nz):
            vertical_divergence[:, :, i] = self.ds[f"S{self.nz - i:0>3}VERTIC.DIVER"][
                ...
            ].T

        return vertical_divergence

    def get_surface_velocities(self) -> Tuple[np.ndarray]:
        return (
            self.ds[f"S{self.nz:0>3}WIND.U.PHYS"][...].T,
            self.ds[f"S{self.nz:0>3}WIND.V.PHYS"][...].T,
        )

    def get_surface_geopotential(self) -> np.ndarray:
        return self.ds["SPECSURFGEOPOTEN"][...].T

    def get_temperature(self) -> np.ndarray:
        temperature = np.zeros((self.nx, self.ny, self.nz))

        for i in range(self.nz):
            temperature[:, :, i] = self.ds[f"S{self.nz - i:0>3}TEMPERATURE"][...].T

        return temperature

    def get_pressure_departure(self) -> np.ndarray:
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

    def get_surface_hyrdostatic_pressure(self):
        return self.ds["SURFPRESSION"][...].T
