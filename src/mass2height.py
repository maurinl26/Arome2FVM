# -*- coding: utf-8 -*-
import numpy as np


def mass2height_coordinates(
    hybrid_coef_A: np.ndarray,
    hybrid_coef_B: np.ndarray,
    surface_pressure: np.ndarray,
    temperature_interfaces: np.ndarray,
    z_surface: np.ndarray,
    Rd: float,
    Rd_cpd: float,
    gravity0: float,
    nx: int,
    ny: int,
    nz: int,
    nz_faces: int,
):

    z_tilde = np.zeros((nx, ny, nz_faces))
    delta_p_rel = np.zeros((nx, ny, nz))

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
    delta_p_rel[:, :, 1:] = delta_p_tilde[:, :, 1:] / p[:, :, 1:]
    delta_p_rel[:, :, 0] = -(1 + 1 / Rd_cpd)

    temperature = np.sqrt(
        temperature_interfaces[:, :, 1:] * temperature_interfaces[:, :, :nz]
    )

    z_temp = (Rd / gravity0) * temperature * delta_p_rel

    # Pourrait être assimilé sur Tools
    z_tilde[:, :, nz] = z_surface
    for i in range(nz, 0, -1):
        z_tilde[:, :, i - 1] = z_tilde[:, :, i] - z_temp[:, :, i - 1]

    alpha = 1 - np.divide(p, p_tilde[:, :, 1:])
    alpha[:, :, 0] = -1

    factor = alpha / delta_p_rel
    zcr = z_tilde[:, :, :nz] * factor + (1 - factor) * z_tilde[:, :, 1:]

    return zcr
