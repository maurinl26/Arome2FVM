# Arome2FVM

Conversion tool from AROME state variables to FVM ones. Among transformations, mass based coordinate is translated to height based terrain following coordinate, vertical wind is computed from vertical divergence, and exner pressure and potential temperature are computed.

## Installation

## EpyGram for raw Arome file .fa to .nc

Conversion from .fa file to .nc file is performed with EPyGram software : https://github.com/UMR-CNRM/EPyGrAM

Installation of EPyGram :
```

```

Command for conversion :
```
epy_conv.py ./files/historic.arome.fa -o nc
```

## Extraction of orography and vertical coordinates

```
python src/app.py ../files/historic.arome.nc ./config/alps.yml ./config/arome.nc
```

## Launch FVM runs

## Calculations
