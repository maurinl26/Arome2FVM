# -*- coding: utf-8 -*-
""" Initialization from AROME initial conditions
    (netCDF4 file converted from a FA file)
    Including orography and vertical coordinate over 90 levels
"""
import numpy as np
from functools import cached_property
from typing import TYPE_CHECKING

from arome_reader import AromeReader
from levels_conversion import mass2height_coordinates

if TYPE_CHECKING:
    from fvms.model.config import Config
    from fvms.utils.storage import Field
    from fvms.utils.typingx import Triple

from functools import cached_property
from typing import TYPE_CHECKING

from fvms.geometry.coordinates import Grid, HorizontalCoordinates
from fvms.model.config import Config
from fvms.utils.storage import Field, to_numpy
from fvms.utils.typingx import Triple


class AROME(Config):

    arome_reader: AromeReader

    ps_ref: float = 101325
    ts_ref: float = 288.15
    t_tropo_ref: float = 220
    gamma: float = -0.0065  # Vertical temperature gradient

    def __init__(self, arome_file: str):
        super().__init__()

        # AROME Reader
        arome_reader = AromeReader(arome_file)

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

        # TODO : refactor
        self.dims = arome_reader.get_dims()

        self.vertical_divergence = arome_reader.get_vertical_divergence()
        self.vel_surface = arome_reader.get_surface_velocities()
        self.surface_geopotential = arome_reader.get_surface_geopotential()

        self.zorog = self.surface_geopotential / self.gravity0

        # Vertical velocity from geopotential
        self.wvel = self.vertical_velocity(
            u_surface=self.vel_surface[0],
            v_surface=self.vel_surface[1],
            surface_geopotential=self.surface_geopotential,
            vertical_divergence=self.vertical_divergence,
        )

        # Other fields
        self.temperature = arome_reader.get_temperature()
        self.pressure = arome_reader.get_pressure()
        self.horizontal_velocities = arome_reader.get_horizontal_velocities()

        self.hybrid_coef_A = arome_reader.get_hybrid_coef_A()
        self.hybrid_coef_B = arome_reader.get_hybrid_coef_B()
        self.surface_pressure = arome_reader.get_surface_pressure()

        # Hydrostatic pressure
        self.hydro_press = self.hydrostatic_pressure(
            hybrid_coef_A=self.hybrid_coef_A,
            hybrid_coef_B=self.hybrid_coef_B,
            surface_pressure=self.surface_pressure,
        )

        # define functions for further use
        self.define_orography = self.define_orography_from_arome
        self.define_vertical_coordinate = self.define_vertical_coordinate
        self.zcr = self.zcr()

    def define_orography_from_arome(
        self, grid: Grid = None, horizontal_coordinates: HorizontalCoordinates = None
    ) -> np.ndarray:
        return self.zorog

    def define_vertical_coordinate(
        self,
        grid: Grid = None,
        zstretch: np.ndarray = None,
        zorog: np.ndarray = None,
        bottom: float = None,
        zorog_smooth: np.ndarray = None,
    ) -> np.ndarray:
        return self.zcr()

    def vertical_velocity(
        self,
        u_surface: np.ndarray,
        v_surface: np.ndarray,
        surface_geopotential: np.ndarray,
        vertical_divergence: np.ndarray,
        dx: float = 1250,
        dy: float = 1250,
    ) -> np.ndarray:
        """Compute vertical velocity from vertical divergence and surface wind.

        Args:
            u_surface (np.ndarray): u wind at surface
            v_surface (np.ndarray): v wind at surface
            surface_geopotential (np.ndarray): geopotential on surface
            vertical_divergence (np.ndarray): vertical divergence on the whole domain
            dx (float, optional): x spacing. Defaults to 1250.
            dy (float, optional): y spacing. Defaults to 1250.
        """

        # Geopotential
        geopo = self.zcr() * self.gravity0
        d_geopo = np.gradient(geopo, axis=2)

        # Surface vertical wind speed
        d_surf_geopo_dx, d_surf_geopo_dy = np.gradient(surface_geopotential, dx, dy)
        w0 = (u_surface * d_surf_geopo_dx + v_surface * d_surf_geopo_dy) / self.gravity0

        # Vertical velocity
        wvel = vertical_divergence / (self.gravity0 * d_geopo)
        wvel[:, :, 0] += w0.T
        wvel = np.cumsum(wvel, axis=2)

        return wvel

    # TODO : replace analytical by numerical computation
    @cached_property
    def hydrostatic_pressure(
        self,
        hybrid_coef_A: np.ndarray,
        hybrid_coef_B: np.ndarray,
        surface_pressure: np.ndarray,
        nz: int = 90,
    ) -> np.ndarray:
        """Compute hydrostatic pressure on cells centers from
        A, B, and surface pressure coefficients.

        Args:
            ds (nc.Dataset): arome historic data
        """
        hydrostatic_pressure_faces = (
            hybrid_coef_A[np.newaxis, np.newaxis, :]
            + surface_pressure.T[:, :, np.newaxis]
            * hybrid_coef_B[np.newaxis, np.newaxis, :]
        )

        # Face to center cells
        hydrostatic_pressure = 0.5 * (
            hydrostatic_pressure_faces[:, :, :nz] + hydrostatic_pressure_faces[:, :, 1:]
        )

        return hydrostatic_pressure

    # TODO : replace analytical by numerical computation
    @cached_property
    def zcr(self) -> np.ndarray:
        """Compute zcr (z coordinates) based on hydrostatic pressure"""
        # Z below tropopause

        z_coordinate = mass2height_coordinates(
            self.hybrid_coef_A,
            self.hybrid_coef_B,
            self.surface_pressure,
            self.temperature,
            self.zorog,
            self.Rd_cpd,
            self.cpd,
            self.gravity0,
            **self.dims,
        )
        return z_coordinate

    ##### Methods used in   thermodynamics.py
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
        rho = self.hydrostatic_pressure / (self.Rd * self.temperature)

    def set_velocity(self, vel: Triple[Field]) -> None:
        # Convert divergence for vertical velocity to vertical velocity
        vel[0] = self.horizontal_velocities[0]
        vel[1] = self.horizontal_velocities[1]
        vel[3] = self.vertical_velocity()
