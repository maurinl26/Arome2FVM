# -*- coding: utf-8 -*-
import numpy as np
import os

# TODO(tehrengruber): either capatalize all names here or put them in a dict
# TODO(tehrengruber): make this configurable using environment variables

# (gt:cpu_ifirst, gt:cpu_kfirst, gt:gpu, numpy, ...)
backend = "numpy"

dtype_float = np.float64

# defaults
backend_opts = {}

if backend.startswith("gt") or backend.startswith("dace"):
    backend_opts = {**backend_opts, "verbose": True}

# directory for caching gt4py stencil sources
# TODO(tehrengruber): the install directory might be read only. Fallback gracefully in that case.
if "GT_CACHE_ROOT" not in os.environ:
    backend_opts["cache_settings"] = {"root_path": os.path.dirname(__file__)}

rebuild = False

# prints a hash of all stencil arguments before and after each stencil call
debug_stencil_calls = False

# compile stencils in parallel (experimental, occasionally breaks)
parallel_compilation = True

# validate domain and origin arguments on stencil call (significant performance boost if disabled)
validate_stencil_args = False

jitflags = {
    "fastmath": True,
    "error_model": "numpy",
    "cache": True,
    "boundscheck": True,
    "parallel": False,
}

update_periodic_layers_using_copy_stencil = False
if backend == "dace:gpu":  # initial experiments showed higher performance
    update_periodic_layers_using_copy_stencil = True
