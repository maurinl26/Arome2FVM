# -*- coding: utf-8 -*-
""" Initialization from AROME initial conditions
    (netCDF4 file converted from a FA file)
    Including orography and vertical coordinate over 90 levels
"""
import numpy as np
from functools import cached_property
from typing import TYPE_CHECKING

import yaml

from arome_reader import AromeReader
from mass2height import mass2height_coordinates
from vertical_velocity import vertical_divergence_to_vertical_velocity

from functools import cached_property
from typing import TYPE_CHECKING

from fvms.utils.storage import Field
from fvms.utils.typingx import Triple


class Arome:

    arome_reader: AromeReader

    # Vertical temperature gradient
    ps_ref: float = 101325
    ts_ref: float = 288.15
    t_tropo_ref: float = 220
    gamma: float = -0.0065

    # Indexing of vertical levels
    # arome_level_order: LevelOrder = LevelOrder(LevelOrder.TOP_TO_BOTTOM)
    # fvm_level_order: LevelOrder = LevelOrder(LevelOrder.BOTTOM_TO_TOP)

    def __init__(self, arome_file: str, config_file: str):

        # AROME Reader
        self.arome_reader = AromeReader(arome_file)

        # Constants (from FVM config)
        self.Rd = self.constants.Rd
        self.gravity0 = self.constants.gravity0
        self.p0 = self.constants.p0
        self.Rd_p0 = self.constants.Rd_p0
        self.cpd = self.constants.cpd
        self.Rd_cpd = self.constants.Rd_cpd

        # Composite constants
        self.Gs = self.ts_ref * self.Rd / self.gravity0
        self.Hs = self.gamma / self.ts_ref

        # nx, ny, nz
        self.dims = self.arome_reader.get_dims()

        # Fields from AROME file
        self.vertical_divergence = self.arome_reader.get_vertical_divergence()
        self.vel_surface = self.arome_reader.get_surface_velocities()
        self.surface_geopotential = self.arome_reader.get_surface_geopotential()

        self.zorog = self.surface_geopotential / self.gravity0

        # Vertical velocity from geopotential
        self.wvel = self.vertical_velocity(
            u_surface=self.vel_surface[0],
            v_surface=self.vel_surface[1],
            surface_geopotential=self.surface_geopotential,
            vertical_divergence=self.vertical_divergence,
        )

        # Other fields
        self.temperature = self.arome_reader.get_temperature()
        self.pressure = self.arome_reader.get_pressure()
        self.horizontal_velocities = self.arome_reader.get_horizontal_velocities()

        self.hybrid_coef_A = self.arome_reader.get_hybrid_coef_A()
        self.hybrid_coef_B = self.arome_reader.get_hybrid_coef_B()
        self.surface_pressure = self.arome_reader.get_surface_pressure()

        # define functions for further use
        self.zcr = self.zcr()

    def vertical_velocity(
        self,
    ) -> np.ndarray:

        vertical_velocity = vertical_divergence_to_vertical_velocity(
            self.zcr,
            self.surface_geopotential,
            self.vel_surface[0],
            self.vel_surface[1],
            self.vertical_divergence,
            self.gravity0,
        )

        return vertical_velocity

    @cached_property
    def zcr(self) -> np.ndarray:
        """Compute z coordinates given mass based coefficients.

        Returns:
            np.ndarray: hei
        """

        z_coordinate = mass2height_coordinates(
            self.hybrid_coef_A,
            self.hybrid_coef_B,
            self.surface_pressure,
            self.temperature,
            self.zorog,
            self.Rd,
            self.Rd_cpd,
            self.gravity0,
            **self.dims,
        )
        return z_coordinate

    ##### Methods used in fvms.initialization.
    #                   thermodynamics.py
    #                   velocity.py
    #                   idealized.py (optional)
    #                       testcase_handle.set_theta(theta)
    #                       testcase_handle.set_rho(rho)
    #                       testcase_handle.set_exner(exner)
    #                       testcase_handle.set_rvapour(rvapour)
    #                       testcase_handle.set_velocity()
    #
    #                   (Optional)
    #                       testcase_handle.add_velocity_disturbance(uvel)
    #                       testcase_handle.add_buoyancy_excess()
    def set_theta(self, theta: Field) -> None:
        theta = self.temperature * (self.p0 / self.pressure) ** (self.Rd_cpd)

    def set_exner(self, exner: Field) -> None:
        exner = (self.pressure / self.p0) ** self.Rd_cpd

    def set_rho(self, rho: Field) -> None:
        rho = self.pressure / (self.Rd * self.temperature)

    def set_velocity(self, vel: Triple[Field]) -> None:
        # Convert divergence for vertical velocity to vertical velocity
        vel[0] = self.horizontal_velocities[0]
        vel[1] = self.horizontal_velocities[1]
        vel[2] = self.vertical_velocity()

    def writer(config_file: str):

        with open(config_file, "rw") as f:
            config_dict = yaml.load(f)

            config_dict.update({})
