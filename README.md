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

- Convert vertical coordinate + orography :

```
python src/convert.py --arome-file ../files/historic.arome.nc --data-file ./config/arome.nc
```

- Plot Z coordinate :

```
 python src/plot.py --data-file ./config/arome.nc --fig-file ./config/zcr.png
```

## Calculations

Conversions from mass based coordinates to height based following ones are implemented following Arome dynamical kernel specifications.

```
Bénard P, Vivoda J, Mašek J, Smolı́ková P, Yessad K, Smith Ch, Brožková R, Geleyn J-F. 2010.
Dynamical kernel of the Aladin–NH spectral limited-area model: Revised formulation and sensitivity
experiments. Q. J. R. Meteorol. Soc. 136: 155–169. DOI:10.1002/qj.522
```
