# Arome2FVM

Conversion tool from AROME state variables to FVM ones. Among transformations, mass based coordinate is translated to height based terrain following coordinate, vertical wind is computed from vertical divergence, and exner pressure and potential temperature are computed.

## Installation

```
git clone git@github.com:maurinl26/Arome2FVM.git
cd $WORKDIR/Arome2FVM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## EpyGram for raw Arome file .fa to .nc

Conversion from .fa file to .nc file is performed with EPyGram software : https://github.com/UMR-CNRM/EPyGrAM

Installation of EPyGram :
```

```

Command for conversion of an .fa file to a .nc file :
```
epy_conv.py ./files/historic.arome.fa -o nc
```

## Extraction of orography and vertical coordinates

```
python src/app.py ../files/historic.arome.nc ./config/arome.nc
```


## Calculations

Conversions from mass based coordinates to height based following ones are implemented following Arome
