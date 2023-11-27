# -*- coding: utf-8 -*-
import numpy as np


def vertical_divergence_to_vertical_velocity(
    zcr: np.ndarray,
    specific_surface_geopotential: np.ndarray,
    u_surface: np.ndarray,
    v_surface: np.ndarray,
    vertical_divergence: np.ndarray,
    alpha: np.ndarray,
    delta_p_rel: np.ndarray,
    gravity0: np.ndarray,
    nz: int,
) -> np.ndarray:
    """Computes vertical velocity from vertical divergence.

    Args:
        zcr (np.ndarray): z coordinates
        specific_surface_geopotential (np.ndarray): surface geopotential
        u_surface (np.ndarray): horizontal velocity (first component)
        v_surface (np.ndarray): horizontal velocity (second component)
        vertical_divergence (np.ndarray): vertical divergence field from AROME file
        alpha (np.ndarray): alpha coefficient
        delta_p_rel (np.ndarray): delta pi coefficient
        gravity0 (np.ndarray): gravity constant
        nz (int): number of vertical levels

    Returns:
        np.ndarray: vertical velocities over the domain
    """

    # geopotential field
    geopo = zcr * gravity0

    d_geopo = np.gradient(geopo, axis=2)

    # W surface
    (
        d_specific_surface_geopotential_dx,
        d_specific_surface_geopotential_dy,
    ) = np.gradient(
        specific_surface_geopotential, dx, dy
    )  # ajouter spacing
    w0 = (
        u_surface * d_specific_surface_geopotential_dx
        + v_surface * d_specific_surface_geopotential_dy
    ) / gravity0

    # Niveaux interfaces
    wvel_tilde = vertical_divergence / (gravity0 * d_geopo)
    wvel_tilde[:, :, 0] = w0
    wvel_tilde = np.cumsum(wvel_tilde, axis=2)

    # Niveaux pleins
    factor = np.divide(alpha, delta_p_rel)
    wvel = np.multiply(wvel_tilde[:, :, :nz], factor) + np.multiply(
        (1 - factor), wvel_tilde[:, :, 1:]
    )

    return wvel
