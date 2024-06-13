# -*- coding: utf-8 -*-
from typing import Tuple
from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import (
    computation,
    PARALLEL,
    FORWARD,
    BACKWARD,
    interval,
    IJ,
    K,
    exp,
    sqrt,
)
import numpy as np


def _alpha(
    p: gtscript.Field["float"], p_faces: gtscript.Field["float"]
) -> gtscript.Field["float"]:
    """Compute alpha coefficient on each level

    Args:
        p (gtscript.Field["float"]): hydrostatic pressure (pi) on mass points
        p_faces (gtscript.Field["float"]): hydrostatic pressure on faces

    Returns:
        gtscript.Field["float"]: alpha coefficient
    """

    with computation(FORWARD):

        with interval(0, 1):
            alpha = -1

        with interval(1, -1):
            alpha = 1 - (p / p_faces)


def _dp_faces_p(
    p: gtscript.Field["float"], delta_p_faces: gtscript.Field["float"], Rd_cpd: float
) -> gtscript.Field["float"]:
    """Compute relative diff between delta hydrostatic pressure and mass point pressure on a cell.

    Args:
        p (gtscript.Field["float"]): hydrostatic pressure at mass point
        delta_p_faces (gtscript.Field["float"]): delta pressure on faces
        Rd_cpd (float): constant for dry air Rd / cpd

    Returns:
        gtscript.Field["float"]: ratio
    """

    from __externals__ import Rd_cpd

    with computation(FORWARD):

        with interval(0, 1):
            delta_p_rel = 1 + 1 / Rd_cpd

        with interval(1, -1):
            delta_p_rel = delta_p_faces / p


def _z_faces(
    z_surface: gtscript.Field[IJ, "float"],
    temperature: gtscript.Field["float"],
    delta_p_p: gtscript.Field["float"],
    Rd: float,
    gravity0: float,
) -> gtscript.Field["float"]:
    """Compute z coordinate on interface levels.

    Args:
        z_surface (gtscript.Field["float"]): orography
        temperature (gtscript.Field["float"]): _description_
        delta_p_p (gtscript.Field["float"]): relative diff of hydrostatic pressure on cell
        Rd (float): ideal gas constant for dry air
        gravity0 (float): gravity constant
        nz (int): n of levels

    Returns:
        gtscript.Field["float"]: z_coordinate on faces
    """
    # TODO : use grid tools from FVM to handle nx, ny, nz

    with computation(PARALLEL), interval(...):
        z_temp = (Rd / gravity0) * temperature * delta_p_p

    with computation(FORWARD):
        with interval(0, 1):
            z_faces = z_surface

        with interval(1, -1):
            z_faces[0, 0, 1] = z_faces[0, 0, 0] - z_temp[0, 0, 1]


def _pressure_from_coeff(
    p_tilde: gtscript.Field["float"],
    hybrid_coef_A: gtscript.Field[K, "float"],
    hybrid_coef_B: gtscript.Field[K, "float"],
    surface_pressure: gtscript.Field[IJ, "float"],
    p: gtscript.Field["float"],
) -> Tuple[gtscript.Field["float"]]:
    """Compute pressure from surface and hybrid coefficients"""

    with computation(FORWARD), interval(...):
        p_tilde = hybrid_coef_A + exp(surface_pressure) * hybrid_coef_B

    with computation(FORWARD), interval(...):
        p[0, 0, 0] = sqrt(p_tilde[0, 0, 0] * p_tilde[0, 0, 1])


def _p2zcr(
    alpha: gtscript.Field["float"],
    delta_p_rel: gtscript.Field["float"],
    z_tilde: gtscript.Field["float"],
    zcr: gtscript.Field["float"],
) -> gtscript.Field["float"]:

    with computation(FORWARD), interval(...):
        factor = alpha / delta_p_rel
        zcr[0, 0, 0] = z_tilde[0, 0, 0] * factor + (1 - factor) * z_tilde[0, 0, 1]


def mass2height_coordinates(
    hybrid_coef_A: gtscript.Field[K, "float"],
    hybrid_coef_B: gtscript.Field[K, "float"],
    surface_pressure: gtscript.Field[IJ, "float"],
    temperature: gtscript.Field["float"],
    z_surface: gtscript.Field["float"],
    alpha: gtscript.Field["float"],
    p: gtscript.Field["float"],
    p_faces: gtscript.Field["float"],
    delta_p_tilde: gtscript.Field["float"],
    delta_p_rel: gtscript.Field["float"],
    p_tilde: gtscript.Field["float"],
    z_tilde: gtscript.Field["float"],
    zcr: gtscript.Field["float"],
) -> gtscript.Field["float"]:
    """Converts mass based coordinate to height based terrain following coordinate.

    Args:
        hybrid_coef_A (gtscript.Field["float"]): A coeff on faces
        hybrid_coef_B (gtscript.Field["float"]): B coeff on faces (linked with surface pressure)
        surface_pressure (gtscript.Field["float"]): surface hydrostatic pressure
        temperature_faces (gtscript.Field["float"]): temperature at mass point
        z_surface (gtscript.Field["float"]): orography
        Rd (float): constant of ideal gas for dry air
        Rd_cpd (float): Rd / cpd
        gravity0 (float): constant of gravity
        nx (int): first horizontal dimension
        ny (int): second horizontal dimension
        nz (int): vertical levels for mass points
        nz_faces (int): vertical levels at faces

    Returns:
        _type_: _description_
    """

    _alpha(p, p_faces)

    # 90 niveaux 0 - 89
    _dp_faces_p(p, delta_p_tilde)

    _z_faces(z_surface, temperature, delta_p_rel)

    _alpha(p, p_tilde)

    _p2zcr(alpha, delta_p_rel, z_tilde, zcr)

    return np.flip(zcr, 2)
