

import os
import sys
sys.path.insert(0, os.path.abspath('C:/Users/Idrissa Belem/Documents/GitHub/morta_nuts2/src'))  # pointe vers src/ C:/Users/Idrissa Belem/Documents/GitHub/morta_nuts2\src

#sys.path.insert(0, os.path.abspath('../../src'))

project = 'Morta-nuts'
copyright = '2026, Innovation'
author = 'Innovation Lab'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

