morta\_nuts2 Documentation
==========================

.. raw:: html

   <div style="text-align: center; margin-bottom: 20px;">
     <img src="_static/LogoDetralytics.png" style="height: 80px; margin-bottom: 10px;"/>
     <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
       <img src="https://img.shields.io/badge/python-3.9%2B-blue"/>
       <img src="https://img.shields.io/badge/doc-Sphinx-orange"/>
       <img src="https://img.shields.io/badge/data-Eurostat-blueviolet"/>
       <img src="https://img.shields.io/badge/env-uv-black"/>
     </div>
   </div>

----

Project Overview
-----------------

**morta_nuts2** is a Python package for modeling and analyzing mortality rates
at the **NUTS2** regional level across Europe, using open data from **Eurostat**.

It is designed for actuaries, demographers and researchers who need sub-national
mortality analysis at the European scale: construction of regional life tables,
fitting of stochastic mortality models, projection of future mortality rates,
and geographic visualization.

.. note::
   The geographic breakdown used is the **NUTS2** reference framework
   (Nomenclature of Territorial Units for Statistics, level 2) provided by
   Eurostat via the shapefile ``NUTS_RG_01M_2024_3035.shp``.

----

Key Features
-------------

- **Download and cache** mortality, deaths, and population data from the Eurostat API.
- **Build regional life tables** from raw data (rates, exposures, deaths).
- **Fit mortality models**: Lee-Carter (LC-p) and Lee & Li (LL-p) with B-splines.
- **Project future mortality rates** at the regional level.
- **Visualize** mortality surfaces and choropleth maps across NUTS2 regions.

----

Installation
-------------

This project uses `uv <https://github.com/astral-sh/uv>`_ to manage the virtual
environment and dependencies. All packages are locked in ``uv.lock`` and declared
in ``pyproject.toml``.

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-organisation/morta_nuts2.git
   cd morta_nuts2

   # Create and activate the virtual environment with uv
   uv venv
   
   # Install all dependencies from the lockfile
   uv sync

   # Or used the package with :
   uv pip install git+https://github.com/Detralytics/morta_nuts2.git
 

.. tip::
   ``uv sync`` reads ``uv.lock`` and installs the exact same package versions
   on every machine, ensuring full reproducibility of the environment.

----

Project Structure
------------------

.. code-block:: text

   morta_nuts2/
   │
   ├── src/
   │   └── morta_nuts2/
   │       ├── data/
   │       │   ├── eurostat.py            # Eurostat API client & data manager
   │       │   └── mortality_table.py     # Mortality table construction
   │       │
   │       ├── graph/
   │       │   └── plot_class.py          # Low-level visualization tools
   │       │
   │       └── model/
   │           ├── Bsplines/
   │           │   └── Bsplines.py        # B-spline basis functions
   │           ├── LC_p/
   │           │   └── lcp_class.py       # Lee-Carter parametric model
   │           ├── LL_p/
   │           │   └── llp_class.py       # Lee & Li parametric model
   │           ├── parameters_init/
   │           │   └── param_init.py      # Parameter initialization
   │           └── projection/
   │               └── project_class.py   # Mortality projection engine
   │
   ├── notebook/                          # Jupyter notebooks for exploration
   ├── NUTS_files/                        # NUTS2 shapefiles (Eurostat)
   ├── pyproject.toml                     # Project metadata & dependencies
   ├── uv.lock                            # Locked dependency versions
   └── docs/                              # Sphinx documentation (this site)

----

API Reference
--------------

Browse the documentation by module. Click on a section to access the full
description of all classes, methods and functions.

.. toctree::
   :maxdepth: 2
   :caption: Data:

   api/data/eurostat
   api/data/mortality_table

.. toctree::
   :maxdepth: 2
   :caption: Models:

   api/model/bsplines
   api/model/lcp
   api/model/llp
   api/model/param_init
   api/model/projection

.. toctree::
   :maxdepth: 2
   :caption: Visualization:

   api/visualisation/plot_class

----

Examples & Notebooks
---------------------

.. note::
   The following notebooks illustrate the usage of the **morta_nuts2** package.
   They are displayed in **read-only mode**: code is shown with its pre-computed
   outputs but cannot be executed from this documentation.

.. toctree::
   :maxdepth: 0
   :caption: Notebooks:

   notebook/exploration_deterministe_BE
   notebook/exploration_stochastique_BE
   notebook/exploration_deterministe_FR
   notebook/exploration_stochastique_FR
----

Indices and Tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`