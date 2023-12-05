# -*- coding: utf-8 -*-
fieldnames_nodes_3d = {
    "vertical_divergence": "VERTIC.DIVER",
    "temperature": "TEMPERATURE",
    "pressure_departure": "PRESS.DEPART",
    "uvel": "WIND.U.PHYS",
    "vvel": "WIND.V.PHYS",
    "tke": "TKE",
    "rsnow": "SNOW",
    "rrain": "RAIN",
    "rgraupel": "GRAUPEL",
    "rliquid": "HUMI.SPECIF",
    "rice": "ICE_CRYSTAL",
}

fieldnames_surface = {
    "surface_geopotential": "SPECSURFGEOPOTEN",
    "surface_hydrostatic_pressure": "SURFPRESSION",
}

fieldnames_vertical_faces = {
    "hybrid_coef_A": "hybrid_coef_A",
    "hybrid_coef_B": "hybrid_coef_B",
}

fieldnames_double = {"vel": ["WIND.U.PHYS", "WIND.V.PHYS"]}
