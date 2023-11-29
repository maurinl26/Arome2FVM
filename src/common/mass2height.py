# -*- coding: utf-8 -*-
import numpy as np


def _alpha(p: np.ndarray, p_faces: np.ndarray) -> np.ndarray:
    """Compute alpha coefficient on each level

    Args:
        p (np.ndarray): hydrostatic pressure (pi) on mass points
        p_faces (np.ndarray): hydrostatic pressure on faces

    Returns:
        np.ndarray: alpha coefficient
    """

    alpha = 1 - (p / p_faces[:, :, 1:])
    alpha[:, :, 0] = -1

    return alpha


def dp_faces_p(p: np.ndarray, delta_p_faces: np.ndarray, Rd_cpd: float) -> np.ndarray:
    """Compute relative diff between delta hydrostatic pressure and mass point pressure on a cell.

    Args:
        p (np.ndarray): hydrostatic pressure at mass point
        delta_p_faces (np.ndarray): delta pressure on faces
        Rd_cpd (float): constant for dry air Rd / cpd

    Returns:
        np.ndarray: ratio
    """
    delta_p_rel = np.zeros(p.shape)
    delta_p_rel[:, :, 1:] = delta_p_faces[:, :, 1:] / p[:, :, 1:]
    delta_p_rel[:, :, 0] = -(1 + 1 / Rd_cpd)

    return delta_p_rel


def z_faces(
    z_surface: np.ndarray,
    temperature: np.ndarray,
    delta_p_p: np.ndarray,
    Rd: float,
    gravity0: float,
    nx: int,
    ny: int,
    nz: int,
) -> np.ndarray:
    """Compute z coordinate on interface levels.

    Args:
        z_surface (np.ndarray): orography
        temperature (np.ndarray): _description_
        delta_p_p (np.ndarray): relative diff of hydrostatic pressure on cell
        Rd (float): ideal gas constant for dry air
        gravity0 (float): gravity constant
        nz (int): n of levels

    Returns:
        np.ndarray: z_coordinate on faces
    """
    z_temp = (Rd / gravity0) * temperature * delta_p_p

    z_faces = np.zeros((nx, ny, nz + 1))
    # Pourrait être assimilé sur Tools
    z_faces[:, :, nz] = z_surface
    for i in range(nz, 0, -1):
        z_faces[:, :, i - 1] = z_faces[:, :, i] - z_temp[:, :, i - 1]

    return z_faces


def mass2height_coordinates(
    hybrid_coef_A: np.ndarray,
    hybrid_coef_B: np.ndarray,
    surface_pressure: np.ndarray,
    temperature: np.ndarray,
    z_surface: np.ndarray,
    Rd: float,
    Rd_cpd: float,
    gravity0: float,
    nx: int,
    ny: int,
    nz: int,
    nz_faces: int,
) -> np.ndarray:
    """Converts mass based coordinate to height based terrain following coordinate.

    Args:
        hybrid_coef_A (np.ndarray): A coeff on faces
        hybrid_coef_B (np.ndarray): B coeff on faces (linked with surface pressure)
        surface_pressure (np.ndarray): surface hydrostatic pressure
        temperature_faces (np.ndarray): temperature at mass point
        z_surface (np.ndarray): orography
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
    p_tilde = (
        hybrid_coef_A[np.newaxis, np.newaxis, :]
        + np.exp(surface_pressure[:, :, np.newaxis])
        * hybrid_coef_B[np.newaxis, np.newaxis, :]
    )

    # 90 niveaux (0 -> 89)
    p = np.sqrt(p_tilde[:, :, 1:] * p_tilde[:, :, :nz])

    delta_p_tilde = p_tilde[:, :, :nz] - p_tilde[:, :, 1:]

    # 90 niveaux 0 - 89
    delta_p_rel = dp_faces_p(p, delta_p_tilde, Rd_cpd)

    z_tilde = z_faces(z_surface, temperature, delta_p_rel, Rd, gravity0, nx, ny, nz)

    alpha = _alpha(p, p_tilde)

    factor = alpha / delta_p_rel
    zcr = z_tilde[:, :, :nz] * factor + (1 - factor) * z_tilde[:, :, 1:]

    return zcr
