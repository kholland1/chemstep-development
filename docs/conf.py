# docs/conf.py ---------------------------------
import os, sys, datetime

sys.path.insert(0, os.path.abspath('..'))          # find chemstep
sys.path.insert(0, os.path.abspath('../src'))
project       = 'ChemSTEP'
copyright     = f'{datetime.datetime.now().year}, Olivier Mailhot'
author        = 'Olivier Mailhot'
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",   # Google-/NumPy-style docstrings
    "sphinx.ext.intersphinx",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": False,
    "show-inheritance": True,
}

suppress_warnings = ['misc.removed-in-sphinx80warning']
html_theme    = 'sphinx_rtd_theme'
