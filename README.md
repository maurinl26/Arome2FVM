# Arome2FVM

Conversion tool from AROME state variables to FVM ones. Among transformations, mass based coordinate is translated to height based terrain following coordinate, vertical wind is computed from vertical divergence, and exner pressure and potential temperature are computed.

## Installation

```bash
    git clone https://github.com:maurinl26/Arome2FVM.git
    cd $WORKDIR/Arome2FVM
    uv venv
    source .venv/bin/activate
```
## EpyGram for raw Arome file .fa to .nc

Conversion from .fa file to .nc file is performed with EPyGram software : https://github.com/UMR-CNRM/EPyGrAM

Command for conversion of an .fa file to a .nc file :
```bash
    uv run epy_conv.py ./files/historic.arome.fa -o nc
```

## Extraction of orography and vertical coordinates

- Convert vertical coordinate + orography :

```bash
    uv run arome2fvm convert-vertical-coordinate \
     --arome-file ../files/historic.arome.nc \
     --data-file ./config/arome.nc
```

- Plot Z coordinate :

```bash
    uv run arome2fvm plot-vertical-coordinate \
        --data-file ./config/arome.nc \
        --fig-file ./config/zcr.png
```

## Implementation in Jax

The code can run on GPUs/TPUs with Jax. [Changes](/JAX_MIGRATION_SUMMARY.md).

## Calculations

Conversions from mass based coordinates to height based following ones are implemented following Arome.
