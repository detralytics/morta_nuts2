# morta_nuts2

A Python package for modeling and analyzing mortality rates at the NUTS2 regional level across Europe, using Eurostat data.

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
├── notebook/                        # Jupyter notebooks for exploration
├── NUTS_files/                      # NUTS2 shapefiles (Eurostat)
└── docs/                            # Sphinx documentation

```

## Documentation

The full API documentation is generated with Sphinx.

```bash
uv add sphinx sphinx-rtd-theme
cd docs
make html
```

Then open `docs/build/html/index.html` in your browser.
