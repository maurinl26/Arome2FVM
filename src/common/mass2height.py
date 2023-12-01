# -*- coding: utf-8 -*-
import numpy as np
from gt4py.cartesian import stencil, Field
from gt4py_config import dtype_float, backend


@stencil(backend=backend)
def _alpha(
    alpha: Field[dtype_float], p: Field[dtype_float], p_faces: Field[dtype_float]
) -> Field[dtype_float]:
    # From top to bottom
    with computation(BACKWARD):
        with interval(0, 1):
            alpha[0, 0, 0] = -1

        with interval(1, -1):
            alpha[0, 0, 0] = 1 - (p_faces[0, 0, -1 / 2] / p_faces[0, 0, 1 / 2])


@stencil(backend=backend)
def _p_faces(
    p_faces: Field[dtype_float],
    hybrid_coef_A: Field[dtype_float],
    hybrid_coef_B: Field[dtype_float],
    surface_pressure: Field[dtype_float],
):
    with computation(FORWARD), interval(...):
        p_faces[0, 0, 0] = hybrid_coef_A[0] + hybrid_coef_B[0] * surface_pressure[0, 0]


@stencil(backend=backend)
def _p_mass(p_faces: Field[dtype_float], p: Field[dtype_float]):
    with computation(FORWARD), interval(...):
        p[0, 0, 0] = sqrt(p_faces[0, 0, -1 / 2] * p_faces[0, 0, 1 / 2])


@stencil(backend=backend)
def _delta(
    delta: Field[dtype_float],
    p: Field[dtype_float],
    p_faces: Field[dtype_float],
    Rd_cpd: dtype_float,
):
    """Compute relative diff between delta hydrostatic pressure and mass point pressure on a cell.

    Args:
        p (Field[dtype_float]): hydrostatic pressure at mass point
        delta_p_faces (Field[dtype_float]): delta pressure on faces
        Rd_cpd (float): constant for dry air Rd / cpd

    Returns:
        Field[dtype_float]: ratio
    """

    with computation(BACKWARD):
        with interval(0, 1):
            delta[0, 0, 0] = 1 + 1 / Rd_cpd
        with interval(1, -1):
            delta[0, 0, 0] = (p_faces[0, 0, 1 / 2] - p_faces[0, 0, -1 / 2]) / p[0, 0, 0]


@stencil(backend=backend)
def _z_faces(
    z_faces: Field[dtype_float],
    z_surface: Field[dtype_float],
    z_temp: Field[dtype_float],
    temperature: Field[dtype_float],
    delta: Field[dtype_float],
    Rd: float,
    gravity0: float,
) -> Field[dtype_float]:
    """Compute z coordinate on interface levels.

    Args:
        z_surface (Field[dtype_float]): orography
        temperature (Field[dtype_float]): _description_
        delta_p_p (Field[dtype_float]): relative diff of hydrostatic pressure on cell
        Rd (float): ideal gas constant for dry air
        gravity0 (float): gravity constant
        nz (int): n of levels

    Returns:
        Field[dtype_float]: z_coordinate on faces
    """
    with computation(FORWARD), interval(...):
        z_temp[0, 0, 0] = (Rd / gravity0) * temperature[0, 0, 0] * delta[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            z_faces[0, 0, 0] = z_surface[0, 0]
        with interval(1, -1):
            z_faces[0, 0, 0] = z_faces[0, 0, -1] - z_temp[0, 0, 0]

    return z_faces


@stencil(backend=backend)
def _z_mass(
    zcr: Field[dtype_float],
    z_faces: Field[dtype_float],
    alpha: Field[dtype_float],
    delta: Field[dtype_float],
):
    with computation(FORWARD), interval(...):
        factor = alpha[0, 0, 0] / delta[0, 0, 0]
        zcr[0, 0, 0] = (
            z_faces[0, 0, -1 / 2] * factor + (1 - factor) * z_faces[0, 0, 1 / 2]
        )


def mass2height_coordinates(
    p: Field[dtype_float],
    p_faces: Field[dtype_float],
    zcr: Field[dtype_float],
    z_faces: Field[dtype_float],
    hybrid_coef_A: Field[dtype_float],
    hybrid_coef_B: Field[dtype_float],
    surface_pressure: Field[dtype_float],
    temperature: Field[dtype_float],
    z_surface: Field[dtype_float],
    z_temp: Field[dtype_float],
    alpha: Field[dtype_float],
    delta: Field[dtype_float],
    Rd: float,
    Rd_cpd: float,
    gravity0: float,
):
    """Converts mass based coordinate to height based terrain following coordinate.

    Args:
        hybrid_coef_A (Field[dtype_float]): A coeff on faces
        hybrid_coef_B (Field[dtype_float]): B coeff on faces (linked with surface pressure)
        surface_pressure (Field[dtype_float]): surface hydrostatic pressure
        temperature_faces (Field[dtype_float]): temperature at mass point
        z_surface (Field[dtype_float]): orography
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

    # 91 niveaux (0 -> 90)

    _p_faces(p_faces, hybrid_coef_A, hybrid_coef_B, surface_pressure)

    _p_mass(p_faces, p)

    _alpha(alpha, p, p_faces)

    _delta(delta, p, p_faces, Rd_cpd)

    _z_faces(z_faces, z_surface, z_temp, temperature, delta, Rd, gravity0)

    _z_mass(zcr, z_faces, alpha, delta)
