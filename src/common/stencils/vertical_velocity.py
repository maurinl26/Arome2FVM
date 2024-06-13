# -*- coding: utf-8 -*-
from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import computation, interval, PARALLEL, FORWARD

def vertical_divergence_to_vertical_velocity(
    zcr: gtscript.Field["float"],
    specific_surface_geopotential: gtscript.Field["float"],
    u_surface: gtscript.Field["float"],
    v_surface: gtscript.Field["float"],
    vertical_divergence: gtscript.Field["float"],
    alpha: gtscript.Field["float"],
    delta_p_rel: gtscript.Field["float"],
    gravity0: gtscript.Field["float"],
    dx: float,
    dy: float,
    nz: int,
) -> gtscript.Field["float"]:
    """Computes vertical velocity from vertical divergence.

    Args:
        zcr (gtscript.Field["float"]): z coordinates
        specific_surface_geopotential (gtscript.Field["float"]): surface geopotential
        u_surface (gtscript.Field["float"]): horizontal velocity (first component)
        v_surface (gtscript.Field["float"]): horizontal velocity (second component)
        vertical_divergence (gtscript.Field["float"]): vertical divergence field from AROME file
        alpha (gtscript.Field["float"]): alpha coefficient
        delta_p_rel (gtscript.Field["float"]): delta pi coefficient
        gravity0 (gtscript.Field["float"]): gravity constant
        nz (int): number of vertical levels

    Returns:
        gtscript.Field["float"]: vertical velocities over the domain
    """

    with computation(PARALLEL), interval(...):

        # geopotential field
        geopo = zcr * gravity0

        d_geopo = _gradient(geopo, axis=2)

        # W surface
        (
        d_specific_surface_geopotential_dx,
        d_specific_surface_geopotential_dy,
        ) = _gradient(specific_surface_geopotential, dx, dy)

        w0 = (
            u_surface * d_specific_surface_geopotential_dx
            + v_surface * d_specific_surface_geopotential_dy
        ) / gravity0

    with computation(FORWARD):

        with interval(0, 1):
            wvel_tilde = w0

        with interval(1, -1):
            wvel_tilde = vertical_divergence / (gravity0 * d_geopo)


    with computation(FORWARD), interval(1, -1):
        wvel_tilde[0, 0, 0] += wvel_tilde[0, 0, -1]


    # Niveaux pleins
    with computation(FORWARD), interval(1, -1):
        factor = alpha / delta_p_rel
        wvel = wvel_tilde[0, 0, 0] * factor + wvel_tilde[0, 0, 1] * (1 - factor)
   


@gtscript.function
def _gradient(geopo: gtscript.Field["float"], dx: float, dy: float) -> gtscript.Field["float"]:
    ...