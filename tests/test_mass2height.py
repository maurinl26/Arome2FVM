# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "src"))

from common.mass2height import mass2height_coordinates
from arome_reader import AromeReader

arome_file = os.path.join(os.getcwd(), "files", "nc_files", "historic.arome.nc")

reader = AromeReader(arome_file)


def test_mass2height():
    z = mass2height_coordinates(
        reader.get_hybrid_coef_A(),
        reader.get_hybrid_coef_B(),
        reader.get_surface_pressure(),
        reader.get_temperature(),
        reader.get_surface_geopotential() / 9.81,
        8.314,
        8.314 / 2.9e-2,
        9.81,
        1013,
        757,
        90,
        91,
    )

    i = np.random.randint(0, 1013)
    j = np.random.randint(0, 757)

    for k in range(1, 90):
        assert z[i, j, k] > z[i, j, k - 1]
