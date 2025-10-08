# -*- coding: utf-8 -*-
import os
import numpy as np
import jax.numpy as jnp
from jax import random

from operators.mass2height import mass2height_coordinates
from arome_reader import AromeReader

arome_file = os.path.join(os.getcwd(), "files", "nc_files", "historic.arome.nc")
reader = AromeReader(arome_file)

def test_mass2height():
    # Convert numpy arrays from reader to JAX arrays
    hybrid_coef_A = jnp.array(reader.get_hybrid_coef_A())
    hybrid_coef_B = jnp.array(reader.get_hybrid_coef_B())
    surface_pressure = jnp.array(reader.get_surface_pressure())
    temperature = jnp.array(reader.get_temperature())
    surface_geopotential = jnp.array(reader.get_surface_geopotential())
    
    z = mass2height_coordinates(
        hybrid_coef_A,
        hybrid_coef_B,
        surface_pressure,
        temperature,
        surface_geopotential / 9.81,
        8.314,
        8.314 / 2.9e-2,
        9.81,
        1013,
        757,
        90,
    )

    i = np.random.randint(0, 1013)
    j = np.random.randint(0, 757)

    for k in range(1, 90):
        assert z[i, j, k] > z[i, j, k - 1]
