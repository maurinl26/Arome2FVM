# -*- coding: utf-8 -*-
""" Initialization from AROME initial conditions
    (netCDF4 file converted from a FA file)
    Including orography and vertical coordinate over 90 levels
"""
import numpy as np
import jax.numpy as jnp
from functools import cached_property

from arome2fvm.arome_reader import AromeReader
from arome2fvm.levels import LevelOrder
from arome2fvm.physical_constants import PhysicalConstants
from operators.mass2height import mass2height_coordinates


class Arome2FVM:

    arome_reader: AromeReader
    physical_constants: PhysicalConstants



    # Indexing of vertical levels
    arome_level_order: LevelOrder = LevelOrder(LevelOrder.TOP_TO_BOTTOM)
    fvm_level_order: LevelOrder = LevelOrder(LevelOrder.BOTTOM_TO_TOP)

    def __init__(self, arome_file: str):

        # AROME Reader
        self.arome_reader = AromeReader(arome_file)

        # nx, ny, nz
        self.dims = self.arome_reader.get_dims()
        self.nz_faces = self.arome_reader.get_nz_faces()


        self.coordinates()

        # Fields from AROME file
        self.vertical_divergence = self.arome_reader.get_vertical_divergence()
        self.vel_surface = self.arome_reader.get_surface_velocities()
        self.surface_geopotential = self.arome_reader.get_surface_geopotential()

        self.zorog = self.surface_geopotential / self.gravity0

        # Other fields
        self.temperature = self.arome_reader.get_temperature()
        self.pressure = self.arome_reader.get_pressure()
        self.horizontal_velocities = self.arome_reader.get_horizontal_velocities()

        self.hybrid_coef_A = self.arome_reader.get_hybrid_coef_A()
        self.hybrid_coef_B = self.arome_reader.get_hybrid_coef_B()
        self.surface_pressure = self.arome_reader.get_surface_pressure()

    def coordinates(self):
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
        # Convert numpy arrays to JAX arrays for the mass2height_coordinates function
        hybrid_coef_A_jax = jnp.array(self.hybrid_coef_A)
        hybrid_coef_B_jax = jnp.array(self.hybrid_coef_B)
        surface_pressure_jax = jnp.array(self.surface_pressure)
        temperature_jax = jnp.array(self.temperature)
        z_surface_jax = jnp.array(self.zorog)
        
        z_coordinate = mass2height_coordinates(
            hybrid_coef_A=hybrid_coef_A_jax,
            hybrid_coef_B=hybrid_coef_B_jax,
            surface_pressure=surface_pressure_jax,
            temperature=temperature_jax,
            z_surface=z_surface_jax,
            Rd=self.physical_constants.Rd,
            Rd_cpd=self.physical_constants.Rd_cpd,
            gravity0=self.physical_constants.gravity0,
            **self.dims
        )
        # Convert result back to numpy array if needed
        return np.array(z_coordinate)
