# -*- coding: utf-8 -*-
from typing import Tuple
import numpy as np
from functools import cached_property


class VerticalDiv2Velocity:
    
    def __init__(self, spacings: Tuple[int], grid: Tuple[int]):
        self.dx, self.dy, self.dz = spacings
        self.nx, self.ny, self.nz = grid
        
    @cached_property
    def geopotential(self):
        return self.zcr * self.gravity0
        
        
    def __call__(self,
                specific_surface_geopotential: np.ndarray,
                gravity0: np.ndarray,
            ):
        
        # TODO : temporaries
        u_surface: np.ndarray
        v_surface: np.ndarray
        vertical_divergence: np.ndarray
        alpha: np.ndarray
        delta_pi_rel: np.ndarray
        
        # geopotential field
        geopo = self.zcr * gravity0

        d_geopo = np.gradient(geopo, axis=2)

        # W surface
        (
            d_specific_surface_geopotential_dx,
            d_specific_surface_geopotential_dy,
        ) = np.gradient(specific_surface_geopotential, self.dx, self.dy)

        w0 = (
            u_surface * d_specific_surface_geopotential_dx
            + v_surface * d_specific_surface_geopotential_dy
        ) / gravity0

        # Niveaux interfaces
        wvel_tilde = vertical_divergence / (gravity0 * d_geopo)
        wvel_tilde[:, :, 0] = w0
        wvel_tilde = np.cumsum(wvel_tilde, axis=2)

        # Niveaux pleins
        factor = np.divide(alpha, delta_pi_rel)
        wvel = np.multiply(wvel_tilde[:, :, :self.nz], factor) + np.multiply(
            (1 - factor), wvel_tilde[:, :, 1:]
        )

        return wvel

