# -*- coding: utf-8 -*-
import jax.numpy as jnp


def vertical_divergence_to_vertical_velocity(
    zcr: jnp.ndarray,
    specific_surface_geopotential: jnp.ndarray,
    u_surface: jnp.ndarray,
    v_surface: jnp.ndarray,
    vertical_divergence: jnp.ndarray,
    alpha: jnp.ndarray,
    delta_p_rel: jnp.ndarray,
    gravity0: jnp.ndarray,
    dx: float,
    dy: float,
    nz: int,
) -> jnp.ndarray:
    """Computes vertical velocity from vertical divergence.

    Args:
        zcr (jnp.ndarray): z coordinates
        specific_surface_geopotential (jnp.ndarray): surface geopotential
        u_surface (jnp.ndarray): horizontal velocity (first component)
        v_surface (jnp.ndarray): horizontal velocity (second component)
        vertical_divergence (jnp.ndarray): vertical divergence field from AROME file
        alpha (jnp.ndarray): alpha coefficient
        delta_p_rel (jnp.ndarray): delta pi coefficient
        gravity0 (jnp.ndarray): gravity constant
        nz (int): number of vertical levels

    Returns:
        jnp.ndarray: vertical velocities over the domain
    """

    # geopotential field
    geopo = zcr * gravity0

    d_geopo = jnp.gradient(geopo, axis=2)

    # W surface
    d_specific_surface_geopotential_dx, d_specific_surface_geopotential_dy = jnp.gradient(
        specific_surface_geopotential, dx, dy
    )

    w0 = (
        u_surface * d_specific_surface_geopotential_dx
        + v_surface * d_specific_surface_geopotential_dy
    ) / gravity0

    # Niveaux interfaces
    wvel_tilde = vertical_divergence / (gravity0 * d_geopo)
    wvel_tilde = wvel_tilde.at[:, :, 0].set(w0)
    wvel_tilde = jnp.cumsum(wvel_tilde, axis=2)

    # Niveaux pleins
    factor = alpha / delta_p_rel
    wvel = wvel_tilde[:, :, :nz] * factor + (1 - factor) * wvel_tilde[:, :, 1:]

    return wvel
