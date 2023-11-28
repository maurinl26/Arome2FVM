# -*- coding: utf-8 -*-
import yaml
from arome2fvm.arome import Arome
import netCDF4 as nc

def write_state(config: Arome, data_file: str):

    rootgrp = nc.Dataset(data_file, "w")

    rootgrp.createDimension("x", config.dims[0])
    rootgrp.createDimension("y", config.dims[1])
    rootgrp.createDimension("z", config.dims[2])

    x_coordinate = rootgrp.createVariable("x", "f8", ("x",))
    y_coordinate = rootgrp.createVariable("y", "f8", ("y",))
    z_coordinate = rootgrp.createVariable("z", "f8", ("z",))

    x_coordinate[:] = config.grid.xc
    y_coordinate[:] = config.grid.yc
    z_coordinate[:] = config.grid.zc

    xcr_coordinate = rootgrp.createVariable("xcr", "f8", ("x", "y"))
    ycr_coordinate = rootgrp.createVariable("ycr", "f8", ("x", "y"))

    xcr_coordinate[:] = config.xcr
    ycr_coordinate[:] = config.ycr

    zcr_coordinate = rootgrp.createVariable("zcr", "f8", ("x", "y", "z"))
    zcr_coordinate[:] = config.zcr

    orog = rootgrp.createVariable("orog", "f8", ("x", "y"))
    orog.unit = "m"

    rootgrp.close()
