# -*- coding: utf-8 -*-
import numpy as np


def mass2height_coordinates(
    hybrid_coef_A: np.ndarray,
    hybrid_coef_B: np.ndarray,
    surface_pressure: np.ndarray,
    temperature: np.ndarray,
    z_surface: np.ndarray,
    Rd: float,
    cpd: float,
    gravity0: float,
    nx: int,
    ny: int,
    nz: int,
):
    """_summary_

    Args:
        hybrid_coef_A (np.ndarray): _description_
        hybrid_coef_B (np.ndarray): _description_
        surface_pressure (np.ndarray): _description_
        temperature (np.ndarray): _description_
        z_surface (np.ndarray): _description_
        nx (int): _description_
        ny (int): _description_
        nz (int): _description_
        Rd (float): _description_
        cpd (float): _description_
        gravity0 (float): _description_

    Returns:
        _type_: _description_
    """
    Rd_cpd = Rd / cpd

    z_tilde = np.zeros((nx, ny, nz + 1))
    delta_p_rel = np.zeros((nx, ny, nz))

    # 91 niveaux (0 -> 90)
    p_tilde = (
        hybrid_coef_A[np.newaxis, np.newaxis, :]
        + np.exp(surface_pressure.T[:, :, np.newaxis])
        * hybrid_coef_B[np.newaxis, np.newaxis, :]
    )

    # 90 niveaux (0 -> 89)
    p = np.sqrt(np.multiply(p_tilde[:, :, 1:], p_tilde[:, :, :nz]))

    delta_p_tilde = p_tilde[:, :, :nz] - p_tilde[:, :, 1:]

    # 90 niveaux 0 - 89
    delta_p_rel[:, :, 1:] = np.divide(delta_p_tilde[:, :, 1:], p[:, :, 1:])
    delta_p_rel[:, :, 0] = -(1 + 1 / Rd_cpd)

    z_temp = (Rd / gravity0) * np.multiply(temperature, delta_p_rel)

    # Pourrait être assimilé sur Tools
    z_tilde[:, :, nz] = z_surface
    for i in range(nz, 0, -1):
        z_tilde[:, :, i - 1] = z_tilde[:, :, i] - z_temp[:, :, i - 1]

    alpha = 1 - np.divide(p, p_tilde[:, :, 1:])
    alpha[:, :, 0] = -1

    factor = np.divide(alpha, delta_p_rel)
    zcr = np.multiply(z_tilde[:, :, :nz], factor) + np.multiply(
        (1 - factor), z_tilde[:, :, 1:]
    )

    return zcr
