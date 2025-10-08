# -*- coding: utf-8 -*-
from typing import Tuple
import jax.numpy as jnp
from jax import lax


def _alpha(p: jnp.ndarray, p_faces: jnp.ndarray) -> jnp.ndarray:
    """Compute alpha coefficient on each level

    Args:
        p (jnp.ndarray): hydrostatic pressure (pi) on mass points
        p_faces (jnp.ndarray): hydrostatic pressure on faces

    Returns:
        jnp.ndarray: alpha coefficient
    """

    alpha = 1 - (p / p_faces[:, :, 1:])
    alpha = alpha.at[:, :, 0].set(-1)

    return alpha


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
        temperature (jnp.ndarray): _description_
        delta_p_p (jnp.ndarray): relative diff of hydrostatic pressure on cell
        Rd (float): ideal gas constant for dry air
        gravity0 (float): gravity constant
        nz (int): n of levels

    Returns:
        jnp.ndarray: z_coordinate on faces
    """
    z_temp = (Rd / gravity0) * temperature * delta_p_p

    # Initialize z_faces array
    z_faces = jnp.zeros((nx, ny, nz + 1))
    z_faces = z_faces.at[:, :, nz].set(z_surface)
    
    # Convert the for loop to a scan operation for JAX compatibility
    # Original: for i in range(nz, 0, -1): z_faces[:, :, i - 1] = z_faces[:, :, i] - z_temp[:, :, i - 1]
    def scan_fn(z_current, z_temp_level):
        z_next = z_current - z_temp_level
        return z_next, z_next
    
    # Process levels from nz-1 down to 0
    z_temp_levels = jnp.flip(z_temp, axis=2)  # Reverse to go from nz-1 to 0
    init_val = z_surface
    
    # Scan through the levels
    _, z_levels = lax.scan(scan_fn, init_val, z_temp_levels)
    
    # Set the computed levels (flip back to original order)
    z_faces = z_faces.at[:, :, :nz].set(jnp.flip(z_levels, axis=2))

    return z_faces


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

    Args:
        hybrid_coef_A (jnp.ndarray): A coeff on faces
        hybrid_coef_B (jnp.ndarray): B coeff on faces (linked with surface pressure)
        surface_pressure (jnp.ndarray): surface hydrostatic pressure
        temperature_faces (jnp.ndarray): temperature at mass point
        z_surface (jnp.ndarray): orography
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
        hybrid_coef_A[jnp.newaxis, jnp.newaxis, :]
        + jnp.exp(surface_pressure[:, :, jnp.newaxis])
        * hybrid_coef_B[jnp.newaxis, jnp.newaxis, :]
    )

    # 90 niveaux (0 -> 89)
    p = jnp.sqrt(p_tilde[:, :, 1:] * p_tilde[:, :, :nz])

    delta_p_tilde = p_tilde[:, :, :nz] - p_tilde[:, :, 1:]

    # 90 niveaux 0 - 89
    delta_p_rel = dp_faces_p(p, delta_p_tilde, Rd_cpd)

    z_tilde = z_faces(z_surface, temperature, delta_p_rel, Rd, gravity0, nx, ny, nz)

    alpha = _alpha(p, p_tilde)

    factor = alpha / delta_p_rel
    zcr = z_tilde[:, :, :nz] * factor + (1 - factor) * z_tilde[:, :, 1:]

    return jnp.flip(zcr, 2)
