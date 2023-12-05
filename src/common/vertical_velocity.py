# -*- coding: utf-8 -*-
from gt4py.cartesian import gtscript
from arome2fvm.gt4py_config import dtype_float


def surface_vertical_velocity(
    u: gtscript.Field[dtype_float],
    v: gtscript.Field[dtype_float],
    zorog: gtscript.Field[dtype_float],
):
    NotImplemented


def vertical_divergence_to_vertical_velocity(
    zcr: gtscript.Field[dtype_float],
    specific_surface_geopotential: gtscript.Field[dtype_float],
    u_surface: gtscript.Field[dtype_float],
    v_surface: gtscript.Field[dtype_float],
    vertical_divergence: gtscript.Field[dtype_float],
    alpha: gtscript.Field[dtype_float],
    delta_p_rel: gtscript.Field[dtype_float],
    gravity0: gtscript.Field[dtype_float],
    dx: float,
    dy: float,
    nz: int,
) -> gtscript.Field[dtype_float]:
    """Computes vertical velocity from vertical divergence.

    Args:
        zcr (gtscript.Field[dtype_float]): z coordinates
        specific_surface_geopotential (gtscript.Field[dtype_float]): surface geopotential
        u_surface (gtscript.Field[dtype_float]): horizontal velocity (first component)
        v_surface (gtscript.Field[dtype_float]): horizontal velocity (second component)
        vertical_divergence (gtscript.Field[dtype_float]): vertical divergence gtscript.Field from AROME file
        alpha (gtscript.Field[dtype_float]): alpha coefficient
        delta_p_rel (gtscript.Field[dtype_float]): delta pi coefficient
        gravity0 (gtscript.Field[dtype_float]): gravity constant
        nz (int): number of vertical levels

    Returns:
        gtscript.Field[dtype_float]: vertical velocities over the domain
    """

    # geopotential gtscript.Field
    geopo = zcr * gravity0

    d_geopo = np.gradient(geopo, axis=2)

    # W surface
    (
        d_specific_surface_geopotential_dx,
        d_specific_surface_geopotential_dy,
    ) = np.gradient(specific_surface_geopotential, dx, dy)

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
