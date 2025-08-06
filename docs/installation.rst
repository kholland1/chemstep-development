============
Installation
============

ChemSTEP is pure Python with optional numba acceleration.

.. code-block:: bash

   # From source
   git clone https://github.com/yourlab/chemstep.git
   cd chemstep
   pip install -e .

Required runtime deps
---------------------
* numpy ≥ 1.23
* numba ≥ 0.58
* rdkit ≥ 2023.09
* (optional) slurm‐utils for cluster execution