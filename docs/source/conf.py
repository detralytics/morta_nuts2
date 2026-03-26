import os
import sys

# Make the source package importable by Sphinx
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
project = 'morta_nuts2'
copyright = '2026, Detralytics Innovation Lab'
author = 'Detralytics Innovation Lab'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Auto-generate doc from docstrings
    'sphinx.ext.napoleon',      # Support Google & NumPy docstring styles
    'sphinx.ext.viewcode',      # Add [source] links to the doc
    'sphinx.ext.autosummary',   # Generate summary tables
    'nbsphinx',                 # Render Jupyter notebooks (read-only)
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Autodoc options ---------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Napoleon settings (for NumPy/Google style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- nbsphinx options --------------------------------------------------------
nbsphinx_execute = 'never'     # Never re-execute notebooks — display pre-computed outputs only

# -- HTML output options -----------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    # KEY SETTING: keeps all sections expanded in the sidebar
    # without needing to click
    'collapse_navigation': False,
    'navigation_depth': 4,
    'titles_only': False,
    'sticky_navigation': True,
}   

import shutil

# Copy specific notebooks into docs/source/notebook/ (flat) before building
notebooks_src = os.path.abspath('../../notebook')
notebooks_dst = os.path.abspath('notebook')

os.makedirs(notebooks_dst, exist_ok=True)

notebooks_to_copy = [
    'Exploration_BE/exploration_deterministe_BE.ipynb',
    'Exploration_BE/exploration_stochastique_BE.ipynb',
    'Exploration_FR/exploration_deterministe_FR.ipynb',
    'Exploration_FR/exploration_stochastique_FR.ipynb',
]

for nb in notebooks_to_copy:
    src = os.path.join(notebooks_src, nb)
    dst = os.path.join(notebooks_dst, os.path.basename(nb))
    shutil.copy(src, dst)