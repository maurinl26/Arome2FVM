# -*- coding: utf-8 -*-
from typing import Tuple
from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import (
    computation,
    PARALLEL,
    FORWARD,
    interval,
    IJK,
    IJ,
    K,
    exp,
    sqrt,
    Field,
)
import numpy as np
from arome2fvm.env import backend

@gtscript.stencil(backend=backend)
def _alpha(
    pi: Field[IJK, np.float64], pi_faces: Field[IJK, np.float64]
) -> Field[IJK, np.float64]:
    """Compute alpha coefficient on each level

    Args:
        pi (Field[IJK, np.float64]): hydrostatic pressure (pi) on mass points
        pi_faces (Field[IJK, np.float64]): hydrostatic pressure on faces

    Returns:
        Field[IJK, np.float64]: alpha coefficient
    """

    with computation(FORWARD):
        with interval(0, 1):
            alpha = -1

        with interval(1, -1):
            alpha = 1 - (pi / pi_faces)


@gtscript.stencil(backend=backend)
def _dp_faces_p(
    pi: Field[IJK, np.float64], 
    delta_pi_faces: Field[IJK, np.float64],
) -> Field[IJK, np.float64]:
    """Compute relative diff between delta hydrostatic pressure and mass point pressure on a cell.

    Args:
        pi (Field[IJK, np.float64]): hydrostatic pressure at mass point
        delta_p_faces (Field[IJK, np.float64]): delta pressure on faces

    Returns:
        Field[IJK, np.float64]: ratio
    """

    from __externals__ import Rd_cpd

    with computation(FORWARD):
        with interval(0, 1):
            delta_pi_rel = 1 + 1 / Rd_cpd

        with interval(1, -1):
            delta_pi_rel = delta_pi_faces / pi


@gtscript.stencil(backend=backend)
def _z_faces(
    z_surface: Field[IJ, np.float64],
    temperature: Field[IJK, np.float64],
    delta_p_p: Field[IJK, np.float64],
) -> Field[IJK, np.float64]:
    """Compute z coordinate on interface levels.

    Args:
        z_surface (Field[IJK, np.float64]): orography
        temperature (Field[IJK, np.float64]): _description_
        delta_p_p (Field[IJK, np.float64]): relative diff of hydrostatic pressure on cell
        Rd (float): ideal gas constant for dry air
        gravity0 (float): gravity constant
        nz (int): n of levels

    Returns:
        Field[IJK, np.float64]: z_coordinate on faces
    """
    # TODO : use grid tools from FVM to handle nx, ny, nz
    from __externals__ import Rd, gravity0

    with computation(PARALLEL), interval(...):
        z_temp = (Rd / gravity0) * temperature * delta_p_p

    with computation(FORWARD):
        with interval(0, 1):
            z_faces = z_surface

        with interval(1, -1):
            z_faces[0, 0, 1] = z_faces[0, 0, 0] - z_temp[0, 0, 1]


@gtscript.stencil(backend=backend)
def _pressure_from_coeff(
    pi_tilde: Field[np.float64],
    hybrid_coef_A: Field[K, np.float64],
    hybrid_coef_B: Field[K, np.float64],
    surface_pressure: Field[IJ, np.float64],
    pi: Field[IJK, np.float64],
) -> Tuple[Field[IJK, np.float64]]:
    """Compute pressure from surface and hybrid coefficients"""

    with computation(FORWARD), interval(...):
        pi_tilde = hybrid_coef_A + exp(surface_pressure) * hybrid_coef_B

    with computation(FORWARD), interval(...):
        pi[0, 0, 0] = sqrt(pi_tilde[0, 0, 0] * pi_tilde[0, 0, 1])


@gtscript.stencil(backend=backend)
def _pi2zcr(
    alpha: Field[IJK, np.float64],
    delta_p_rel: Field[IJK, np.float64],
    z_tilde: Field[IJK, np.float64],
    zcr: Field[IJK, np.float64],
) -> Field[IJK, np.float64]:
    with computation(FORWARD), interval(...):
        factor = alpha / delta_p_rel
        zcr[0, 0, 0] = z_tilde[0, 0, 0] * factor + (1 - factor) * z_tilde[0, 0, 1]

