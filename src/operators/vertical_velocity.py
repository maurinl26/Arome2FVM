# -*- coding: utf-8 -*-
"""
Vertical velocity calculation from vertical divergence using JAX.

This module provides JAX-accelerated functions for computing vertical velocity
from vertical divergence in terrain-following coordinates.
"""
import jax.numpy as jnp
from jax import jit


@jit
def vertical_divergence_to_vertical_velocity(
    zcr: jnp.ndarray,
    specific_surface_geopotential: jnp.ndarray,
    u_surface: jnp.ndarray,
    v_surface: jnp.ndarray,
    vertical_divergence: jnp.ndarray,
    alpha: jnp.ndarray,
    delta_p_rel: jnp.ndarray,
    gravity0: float,
    dx: float,
    dy: float,
    nz: int,
) -> jnp.ndarray:
    """Computes vertical velocity from vertical divergence.

    This function uses JAX for GPU acceleration and automatic differentiation.

    Args:
        zcr (jnp.ndarray): z coordinates at mass points (nx, ny, nz)
        specific_surface_geopotential (jnp.ndarray): surface geopotential (nx, ny)
        u_surface (jnp.ndarray): horizontal velocity u-component at surface (nx, ny)
        v_surface (jnp.ndarray): horizontal velocity v-component at surface (nx, ny)
        vertical_divergence (jnp.ndarray): vertical divergence field from AROME (nx, ny, nz+1)
        alpha (jnp.ndarray): alpha coefficient (nx, ny, nz)
        delta_p_rel (jnp.ndarray): delta pi coefficient (nx, ny, nz)
        gravity0 (float): gravity constant (9.81 m/s²)
        dx (float): horizontal grid spacing in x-direction (m)
        dy (float): horizontal grid spacing in y-direction (m)
        nz (int): number of vertical levels

    Returns:
        jnp.ndarray: vertical velocities over the domain (nx, ny, nz)
    """
    # Compute geopotential field
    geopo = zcr * gravity0
    
    # Compute vertical gradient of geopotential
    # Using central differences for the gradient along axis 2
    d_geopo = jnp.gradient(geopo, axis=2)
    
    # Compute surface vertical velocity from horizontal flow over orography
    # Surface geopotential gradients
    d_specific_surface_geopotential_dx, d_specific_surface_geopotential_dy = jnp.gradient(
        specific_surface_geopotential, dx, dy
    )
    
    # Surface vertical velocity (w0)
    w0 = (
        u_surface * d_specific_surface_geopotential_dx
        + v_surface * d_specific_surface_geopotential_dy
    ) / gravity0
    
    # Compute vertical velocity on interface levels (faces)
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    wvel_tilde = vertical_divergence / (gravity0 * d_geopo + epsilon)
    
    # Set surface level
    wvel_tilde = wvel_tilde.at[:, :, 0].set(w0)
    
    # Cumulative sum for integration
    wvel_tilde = jnp.cumsum(wvel_tilde, axis=2)
    
    # Interpolate from faces to mass levels
    # Avoid division by zero in the factor computation
    factor = jnp.divide(alpha, delta_p_rel + epsilon)
    
    # Interpolate: w = factor * w_tilde[k] + (1 - factor) * w_tilde[k+1]
    wvel = jnp.multiply(wvel_tilde[:, :, :nz], factor) + jnp.multiply(
        (1 - factor), wvel_tilde[:, :, 1:]
    )
    
    return wvel
