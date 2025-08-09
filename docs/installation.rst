============
Installation
============

.. important::

    Starting with version **1.0.0** ChemSTEP is distributed on **PyPI**.
    We strongly recommend installing via `pip` whenever possible.
    If you need the latest development snapshot you can still install directly
    from the source distribution tarball (`chemstep‑X.Y.Z.tar.gz`).

Quick install:

.. code-block:: bash

    pip install chemstep

Or install a specific release:

.. code-block:: bash

    pip install chemstep==1.0.0

Installing from a source tarball:

While we wait for the PyPI version to come online (as soon as the preprint is out), you can install ChemSTEP
to your environment on wynton like this:

.. code-block:: bash

    pip install /wynton/group/bks/work/omailhot/chemstep-0.2.1.tar.gz

Dependencies
============

ChemSTEP is written in **pure Python ≥3.9** and has a very small
runtime footprint:

* `numpy` ≥ 1.23
* `numba` ≥ 0.54

All mandatory dependencies are pulled in automatically by `pip`.

If you want to be able to build fingerprint libraries, you will also need:

* `rdkit` ≥ 2022.09

After installation, the following one‑liner should succeed and print the
installed version:

.. code-block:: bash

    python -c "import chemstep, sys; print(chemstep.version())"

Typical output::

    0.2.0

Read next:

* :doc:`manual_docking` –run your first ChemSTEP search.
* :doc:`fingerprint_library` –create your own FpLibrary.
