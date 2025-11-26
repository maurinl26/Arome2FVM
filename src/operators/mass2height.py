# -*- coding: utf-8 -*-
"""
Mass to height coordinate transformation using JAX.

This module provides JAX-accelerated functions for converting from mass-based
coordinates to height-based terrain-following coordinates.
"""
import jax.numpy as jnp
from jax import jit
import jax.lax as lax


@jit
def _alpha(p: jnp.ndarray, p_faces: jnp.ndarray) -> jnp.ndarray:
    """Compute alpha coefficient on each level.

    Args:
        p (jnp.ndarray): hydrostatic pressure (pi) on mass points
        p_faces (jnp.ndarray): hydrostatic pressure on faces

    Returns:
        jnp.ndarray: alpha coefficient
    """
    alpha = 1 - (p / p_faces[:, :, 1:])
    # Use at index update instead of in-place assignment
    alpha = alpha.at[:, :, 0].set(-1.0)
    
    return alpha


@jit
def dp_faces_p(p: jnp.ndarray, delta_p_faces: jnp.ndarray, Rd_cpd: float) -> jnp.ndarray:
    """Compute relative diff between delta hydrostatic pressure and mass point pressure on a cell.

    Args:
        p (jnp.ndarray): hydrostatic pressure at mass point
        delta_p_faces (jnp.ndarray): delta pressure on faces
        Rd_cpd (float): constant for dry air Rd / cpd

    Returns:
        jnp.ndarray: ratio
    """
    delta_p_rel = jnp.zeros(p.shape)
    delta_p_rel = delta_p_rel.at[:, :, 1:].set(delta_p_faces[:, :, 1:] / p[:, :, 1:])
    delta_p_rel = delta_p_rel.at[:, :, 0].set(-(1 + 1 / Rd_cpd))
    
    return delta_p_rel


@jit
def z_faces(
    z_surface: jnp.ndarray,
    temperature: jnp.ndarray,
    delta_p_p: jnp.ndarray,
    Rd: float,
    gravity0: float,
    nx: int,
    ny: int,
    nz: int,
) -> jnp.ndarray:
    """Compute z coordinate on interface levels.

    Args:
        z_surface (jnp.ndarray): orography
        temperature (jnp.ndarray): temperature field
        delta_p_p (jnp.ndarray): relative diff of hydrostatic pressure on cell
        Rd (float): ideal gas constant for dry air
        gravity0 (float): gravity constant
        nx (int): first horizontal dimension
        ny (int): second horizontal dimension
        nz (int): number of vertical levels

    Returns:
        jnp.ndarray: z_coordinate on faces
    """
    z_temp = (Rd / gravity0) * temperature * delta_p_p
    
    # Initialize z_faces array
    z_faces_arr = jnp.zeros((nx, ny, nz + 1))
    z_faces_arr = z_faces_arr.at[:, :, nz].set(z_surface)
    
    # Use lax.fori_loop for efficient reverse iteration
    def body_fn(i, z_arr):
        # i goes from 0 to nz-1, but we want to go from nz to 1
        k = nz - i
        return z_arr.at[:, :, k - 1].set(z_arr[:, :, k] - z_temp[:, :, k - 1])
    
    z_faces_arr = lax.fori_loop(0, nz, body_fn, z_faces_arr)
    
    return z_faces_arr


@jit
def mass2height_coordinates(
    hybrid_coef_A: jnp.ndarray,
    hybrid_coef_B: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    temperature: jnp.ndarray,
    z_surface: jnp.ndarray,
    Rd: float,
    Rd_cpd: float,
    gravity0: float,
    nx: int,
    ny: int,
    nz: int,
) -> jnp.ndarray:
    """Converts mass based coordinate to height based terrain following coordinate.

    This function uses JAX for GPU acceleration and automatic differentiation.

    Args:
        hybrid_coef_A (jnp.ndarray): A coefficient on faces
        hybrid_coef_B (jnp.ndarray): B coefficient on faces (linked with surface pressure)
        surface_pressure (jnp.ndarray): surface hydrostatic pressure (log scale)
        temperature (jnp.ndarray): temperature at mass points
        z_surface (jnp.ndarray): orography (surface height)
        Rd (float): constant of ideal gas for dry air (287.0 J/kg/K)
        Rd_cpd (float): Rd / cpd ratio
        gravity0 (float): constant of gravity (9.81 m/s²)
        nx (int): first horizontal dimension
        ny (int): second horizontal dimension
        nz (int): vertical levels for mass points

    Returns:
        jnp.ndarray: height coordinates at mass points (nx, ny, nz)
    """
    # Compute pressure on faces (nz+1 levels)
    p_tilde = (
        hybrid_coef_A[jnp.newaxis, jnp.newaxis, :]
        + jnp.exp(surface_pressure[:, :, jnp.newaxis])
        * hybrid_coef_B[jnp.newaxis, jnp.newaxis, :]
    )
    
    # Compute pressure at mass points (nz levels)
    p = jnp.sqrt(p_tilde[:, :, 1:] * p_tilde[:, :, :nz])
    
    # Compute pressure differences
    delta_p_tilde = p_tilde[:, :, :nz] - p_tilde[:, :, 1:]
    
    # Compute relative pressure differences
    delta_p_rel = dp_faces_p(p, delta_p_tilde, Rd_cpd)
    
    # Compute heights on faces
    z_tilde = z_faces(z_surface, temperature, delta_p_rel, Rd, gravity0, nx, ny, nz)
    
    # Compute alpha coefficient
    alpha = _alpha(p, p_tilde)
    
    # Interpolate from faces to mass points
    factor = alpha / delta_p_rel
    zcr = z_tilde[:, :, :nz] * factor + (1 - factor) * z_tilde[:, :, 1:]
    
    # Flip vertical axis (heights increase upward)
    return jnp.flip(zcr, axis=2)
