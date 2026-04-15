# morta_nuts2

.. raw:: html

   <div style="text-align: center; margin-bottom: 20px;">
     <img src="docs/source/LogoDetralytics.png" style="height: 80px; margin-bottom: 10px;"/>
     <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
       <img src="https://img.shields.io/badge/python-3.11%2B-blue"/>
       <img src="https://img.shields.io/badge/doc-Sphinx-orange"/>
       <img src="https://img.shields.io/badge/data-Eurostat-blueviolet"/>
       <img src="https://img.shields.io/badge/env-uv-black"/>
     </div>
   </div>

A Python package for modeling and analyzing mortality rates at the NUTS2 regional level across Europe, using Eurostat data.

## Installation

If you are using 'uv' python package manager, you can install the package in your python .venv using the following command:

```
uv pip install git+https://github.com/Detralytics/morta_nuts2.git
```

## Documentation

The full API documentation is generated with Sphinx and is made available via ReadtheDocs at the following link:

https://morta-nuts2.readthedocs.io/en/latest/


## Overview

`morta_nuts2` provides tools to:

- **Download and cache** mortality, deaths, and population data from the Eurostat API
- **Fit mortality models** including Lee-Carter (LC) and Lee and Li (LL) with B-splines
- **Project future mortality rates** at the regional level
- **Visualize** mortality surfaces and geographic distributions across NUTS2 regions

The package is designed for actuarial and demographic research requiring sub-national mortality analysis at the European NUTS2 level.

## Project Structure

```
morta_nuts2/
│
├── src/
│   └── morta_nuts2/
│       ├── data/
│       │   ├── eurostat.py          # Eurostat API client & data manager
│       │   └── mortality_table.py   # Mortality table construction
│       │
│       ├── graph/
│       │   └── plot_class.py              # Visualization tools
│       │
│       └── model/
│           ├── Bsplines/
│           │   └── Bsplines.py      # B-spline basis functions
│           ├── LC_p/
│           │   └── lcp_class.py     # Lee-Carter parametric model
│           ├── LL_p/
│           │   └── llp_class.py     # Lee and Li parametric model
│           ├── parameters_init/
│           │   └── param_init.py    # Parameter initialization
│           └── projection/
│               └── project_class.py # Mortality projection engine
│
├── notebook/                        # Jupyter notebooks for exploration - can be used as tutorials
├── NUTS_files/                      # NUTS2 shapefiles (Eurostat)
└── docs/                            # Sphinx documentation

```

## Disclaimer

This package is provided "as is" for computational purposes only. While we strive for accuracy, the authors and the company accept no liability for errors, unexpected behaviour, or decisions made based on its outputs.
Users are solely responsible for validating results and ensuring their appropriate use in any professional or regulatory context. This tool is not a substitute for qualified actuarial judgement.
