import numpy as np

from constants import Constants

def vertical_velocity(
    zcr: np.ndarray,
    specific_surface_geopotential: np.ndarray,
    u_surface: np.ndarray,
    v_surface: np.ndarray,
    vertical_divergence: np.ndarray,
    gravity0: np.ndarray,
) -> np.ndarray:
    """Compute vertical velocity component from vertical divergence

    Args:
        cst (Constants): _description_
        zcr (np.ndarray): _description_
        specific_surface_geopotential (np.ndarray): _description_
        u_surface (np.ndarray): _description_
        v_surface (np.ndarray): _description_
        vertical_divergence (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """

    # geopotential field
    geopo = zcr * gravity0
        
    d_geopo = np.gradient(geopo, axis=2)

    d_specific_surface_geopotential_dx, d_specific_surface_geopotential_dy = np.gradient(specific_surface_geopotential, dx, dy)  # ajouter spacing
    w0 = (u_surface * d_specific_surface_geopotential_dx + v_surface * d_specific_surface_geopotential_dy) / gravity0

    wvel = vertical_divergence / (gravity0 * d_geopo)
    wvel = np.cumsum(wvel, axis=2)
        
    return w0, wvel
    
    