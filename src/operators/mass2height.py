# -*- coding: utf-8 -*-
from typing import Tuple
from gt4py.cartesian.gtscript import (
    computation,
    PARALLEL,
    FORWARD,
    interval,
    IJ,
    K,
    exp,
    sqrt, 
    Field,
    stencil
)
import numpy as np
from arome2fvm.env import backend


@stencil(backend=backend)
def _alpha(
    pi: Field[np.float64], pi_faces: Field[np.float64]
) -> Field[np.float64]:
    """Compute alpha coefficient on each level

    Args:
        pi (Field[np.float64]): hydrostatic pressure (pi) on mass points
        pi_faces (Field[np.float64]): hydrostatic pressure on faces

    Returns:
        Field[np.float64]: alpha coefficient
    """

    with computation(FORWARD):

        with interval(0, 1):
            alpha = -1

        with interval(1, -1):
            alpha = 1 - (pi / pi_faces)


@stencil(backend=backend)
def _dp_faces_p(
    pi: Field[np.float64], delta_pi_faces: Field[np.float64],
) -> Field[np.float64]:
    """Compute relative diff between delta hydrostatic pressure and mass point pressure on a cell.

    Args:
        p (Field[np.float64]): hydrostatic pressure at mass point
        delta_p_faces (Field[np.float64]): delta pressure on faces
        Rd_cpd (float): constant for dry air Rd / cpd

    Returns:
        Field[np.float64]: ratio
    """

    from __externals__ import Rd_cpd

    with computation(FORWARD):

        with interval(0, 1):
            delta_p_rel = 1 + 1 / Rd_cpd

        with interval(1, -1):
            delta_p_rel = delta_pi_faces / pi


@stencil(backend=backend)
def _z_faces(
    z_surface: Field[IJ, np.float64],
    temperature: Field[np.float64],
    delta_p_p: Field[np.float64],
    Rd: float,
    gravity0: float,
) -> Field[np.float64]:
    """Compute z coordinate on interface levels.

    Args:
        z_surface (Field[np.float64]): orography
        temperature (Field[np.float64]): _description_
        delta_p_p (Field[np.float64]): relative diff of hydrostatic pressure on cell
        Rd (float): ideal gas constant for dry air
        gravity0 (float): gravity constant
        nz (int): n of levels

    Returns:
        Field[np.float64]: z_coordinate on faces
    """
    # TODO : use grid tools from FVM to handle nx, ny, nz

    with computation(PARALLEL), interval(...):
        z_temp = (Rd / gravity0) * temperature * delta_p_p

    with computation(FORWARD):
        with interval(0, 1):
            z_faces = z_surface

        with interval(1, -1):
            z_faces[0, 0, 1] = z_faces[0, 0, 0] - z_temp[0, 0, 1]


@stencil(backend=backend)
def _pressure_from_coeff(
    p_tilde: Field[np.float64],
    hybrid_coef_A: Field[K, np.float64],
    hybrid_coef_B: Field[K, np.float64],
    surface_pressure: Field[IJ, np.float64],
    p: Field[np.float64],
) -> Tuple[Field[np.float64]]:
    """Compute pressure from surface and hybrid coefficients"""

    with computation(FORWARD), interval(...):
        p_tilde = hybrid_coef_A + exp(surface_pressure) * hybrid_coef_B

    with computation(FORWARD), interval(...):
        p[0, 0, 0] = sqrt(p_tilde[0, 0, 0] * p_tilde[0, 0, 1])


@stencil(backend=backend)
def _p2zcr(
    alpha: Field[np.float64],
    delta_p_rel: Field[np.float64],
    z_tilde: Field[np.float64],
    zcr: Field[np.float64],
) -> Field[np.float64]:

    with computation(FORWARD), interval(...):
        factor = alpha / delta_p_rel
        zcr[0, 0, 0] = z_tilde[0, 0, 0] * factor + (1 - factor) * z_tilde[0, 0, 1]


class Mass2HeightCoordinates:
    
    def __init__(self, grid: Tuple[int] = (50, 50, 90)):
        
        self.nx, self.ny, self.nz = grid
    
    def __call__(
        hybrid_coef_A: Field[K, np.float64],
        hybrid_coef_B: Field[K, np.float64],
        surface_pressure: Field[IJ, np.float64],
        temperature: Field[np.float64],
        z_surface: Field[np.float64],
        zcr: Field[np.float64],
    ) -> Field[np.float64]:
        
        
        # TODO : temporaries
        alpha: Field[np.float64]
        pi: Field[np.float64]
        pi_faces: Field[np.float64]
        delta_pi_tilde: Field[np.float64]
        delta_pi_rel: Field[np.float64]
        pi_tilde: Field[np.float64]
        z_tilde: Field[np.float64]
   

        _alpha(pi, pi_faces)

        # 90 niveaux 0 - 89
        _dp_faces_p(pi, delta_pi_tilde)

        _z_faces(z_surface, temperature, delta_pi_rel)

        _alpha(pi, pi_tilde)

        _p2zcr(alpha, delta_pi_rel, z_tilde, zcr)

        return zcr
