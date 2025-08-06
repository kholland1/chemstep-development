# docs/conf.py ---------------------------------
import os, sys, datetime

sys.path.insert(0, os.path.abspath('..'))          # find chemstep
sys.path.insert(0, os.path.abspath('../src'))
project       = 'ChemSTEP'
copyright     = f'{datetime.datetime.now().year}, Olivier Mailhot'
author        = 'Olivier Mailhot'
extensions    = [
    'sphinx.ext.autodoc',          # pull docstrings
    'sphinx.ext.napoleon',         # Google-style docstrings
    'sphinx.ext.viewcode',
]
suppress_warnings = ['misc.removed-in-sphinx80warning']
html_theme    = 'sphinx_rtd_theme'
autodoc_default_options = {'members': True, 'inherited-members': True}
