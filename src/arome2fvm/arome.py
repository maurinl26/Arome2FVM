# -*- coding: utf-8 -*-
""" Initialization from AROME initial conditions
    (netCDF4 file converted from a FA file)
    Including orography and vertical coordinate over 90 levels
"""
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from arome2fvm.arome_reader import AromeReader
from arome2fvm.levels import LevelOrder
from common.numpy_stencils.mass2height import mass2height_coordinates
from dataclasses import asdict, dataclass, field

@dataclass
class Constants:

    # Vertical temperature gradient
    Rd: float = 287.059674
    p0: float = 1000.0e2
    gravity0: float = 9.80665
    cpd: float = 1004.709
    Rd_cpd: float = field(init=False)

    def __post_init__(self):
        self.Rd_cpd = self.Rd / self.cpd

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}



class Arome:

    arome_reader: AromeReader
    constants: Constants

    # Indexing of vertical levels
    arome_level_order: LevelOrder = LevelOrder(LevelOrder.TOP_TO_BOTTOM)
    fvm_level_order: LevelOrder = LevelOrder(LevelOrder.BOTTOM_TO_TOP)

    def __init__(self, arome_file: str):

        # AROME Reader
        self.arome_reader = AromeReader(arome_file)
        self.constants = Constants()

        # nx, ny, nz
        self.dims = self.arome_reader.get_dims()
        self.nz_faces = self.arome_reader.get_nz_faces()

        self.build_horizontal_coordinates()

        # Fields from AROME file
        self.vertical_divergence = self.arome_reader.get_vertical_divergence()
        self.vel_surface = self.arome_reader.get_surface_velocities()
        self.surface_geopotential = self.arome_reader.get_surface_geopotential()

        self.zorog = self.surface_geopotential / self.gravity0

        # Other fields
        self.temperature = self.arome_reader.get_temperature()
        self.pressure_departure = self.arome_reader.get_pressure_departure()
        self.horizontal_velocities = self.arome_reader.get_horizontal_velocities()

        self.hybrid_coef_A = self.arome_reader.get_hybrid_coef_A()
        self.hybrid_coef_B = self.arome_reader.get_hybrid_coef_B()
        self.surface_pressure = self.arome_reader.get_surface_hyrdostatic_pressure()

    def build_horizontal_coordinates(self):
        """Computes horizontal coordinates (cartesian)
        xc, yc : 1d arrays
        xcr, ycr: 2d arrays to map the domain (grid)
        """

        xmin = 0 - self.dims["nx"] * self.arome_reader.get_spacing()[0] / 2
        xmax = 0 + self.dims["nx"] * self.arome_reader.get_spacing()[0] / 2

        ymin = 0 - self.dims["ny"] * self.arome_reader.get_spacing()[1] / 2
        ymax = 0 + self.dims["ny"] * self.arome_reader.get_spacing()[1] / 2

        self.xc = np.linspace(xmin, xmax, self.dims["nx"])
        self.yc = np.linspace(ymin, ymax, self.dims["ny"])

        self.xcr, self.ycr = np.meshgrid(self.xc, self.yc, indexing="ij")

    @cached_property
    def zcr(self) -> np.ndarray:
        """Compute z coordinates given mass based coefficients.

        Returns:
            np.ndarray: hei
        """
        z_coordinate = mass2height_coordinates(
            hybrid_coef_A=self.hybrid_coef_A,
            hybrid_coef_B=self.hybrid_coef_B,
            surface_pressure=self.surface_pressure,
            temperature=self.temperature,
            z_surface=self.zorog,
            Rd=self.Rd,
            Rd_cpd=self.Rd_cpd,
            gravity0=self.gravity0,
            **self.dims
        )
        return z_coordinate

