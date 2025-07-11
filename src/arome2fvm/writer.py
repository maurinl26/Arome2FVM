# -*- coding: utf-8 -*-
from arome2fvm.arome import Arome
import netCDF4 as nc


def write_state(arome2fvm: Arome, data_file: str):

    rootgrp = nc.Dataset(data_file, "w")
    
    rootgrp.createDimension("x", arome2fvm.dims["nx"])
    rootgrp.createDimension("y", arome2fvm.dims["ny"])
    rootgrp.createDimension("z", arome2fvm.dims["nz"])

    x_coordinate = rootgrp.createVariable("x", "f8", ("x",))
    y_coordinate = rootgrp.createVariable("y", "f8", ("y",))

    x_coordinate[:] = arome2fvm.xc
    y_coordinate[:] = arome2fvm.yc

    xcr_coordinate = rootgrp.createVariable("xcr", "f8", ("x", "y"))
    ycr_coordinate = rootgrp.createVariable("ycr", "f8", ("x", "y"))

    xcr_coordinate[:] = arome2fvm.xcr
    ycr_coordinate[:] = arome2fvm.ycr

    zcr_coordinate = rootgrp.createVariable("zcr", "f8", ("x", "y", "z"))
    zcr_coordinate[:] = arome2fvm.zcr

    orog = rootgrp.createVariable("orog", "f8", ("x", "y"))
    orog[:] = arome2fvm.zorogs
    orog.unit = "m"

    rootgrp.close()
