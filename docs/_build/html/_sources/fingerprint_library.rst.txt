===========================
Creating your own FpLibrary
===========================

This guide shows how to build an :class:`~chemstep.fp_library.FpLibrary` object
from raw SMILES files using the built-in :func:`~chemstep.fingerprints.compute_morgan_fps`
function in ChemSTEP.

Overview
--------

An :class:`~chemstep.fp_library.FpLibrary` represents a large, chunked collection
of molecules in both SMILES and binary fingerprint form, with matching integer IDs.
You should generate the fingerprints using ChemSTEP’s high-performance
:func:`~chemstep.fingerprints.compute_morgan_fps` helper, which wraps RDKit and
returns contiguous ``uint8`` NumPy arrays.
For this, you will need to have RDKit installed (see https://www.rdkit.org/docs/Install.html ).

We recommend **not compressing** either the SMILES files or the fingerprint
files:

* Uncompressed SMILES (``.smi``) files are much faster (10-20x) to read.
* Fingerprint files (128 bytes per compound for 1024-bit fingerprints) take up more space
  than the SMILES anyway, and compression would slow down I/O significantly in large-scale runs.

Prerequisites
-------------

* One SMILES file per library chunk (these can be the input to
  :func:`~chemstep.fingerprints.compute_morgan_fps`).
* Corresponding 64bit-integer IDs (e.g., ZINC IDs) in ``.npy`` files.

Workflow
--------

1. **Generate fingerprints from SMILES**

   Using ChemSTEP’s built-in helper:

   .. code-block:: python

      from chemstep.fingerprints import compute_morgan_fps
      import numpy as np

      def write_one_fp_file(smi_file, fp_filename):
          fingerprints = compute_morgan_fps(smi_file)
          np.save(fp_filename, fingerprints)


   This produces one ``.npy`` fingerprint file per input ``.smi`` file.

   The speed is approximately 1000 fingerprints / second on a modern CPU,
   so you will need to parallelize this step for large libraries.


2. **Prepare ID files**
   You should prepare corresponding
   ``.npy`` files containing a 64-bit integer ID for every compound.
   The IDs should be in the same order as the
   SMILES and fingerprints, and should be 64-bit integers (``np.int64``).
   Save matching ``.npy`` files with ``np.int64`` IDs
   (same order as the SMILES/fingerprints). These will be split in chunks the
   same way as the smiles and fingerprints.

3. **Instantiate an FpLibrary**
   Here is an example of how to instantiate an :class:`~chemstep.fp_library.FpLibrary`.

   .. code-block:: python

      from glob import glob
      from chemstep.fp_library import FpLibrary

      lib = FpLibrary(
          fingerprint_files = sorted(glob('absolute/path/to/your/fps*.npy')),
          id_files          = sorted(glob('absolute/path/to/your/zids*.npy')),
          smi_files         = sorted(glob('absolute/path/to/your/smi*.smi')),
          outname           = 'zinc_fplib.pickle'
      )

   The .pickle library file can then be copied anywhere and loaded in lieue of
   rebuilding the library with:

    .. code-block:: python

        from chemstep.fp_library import FpLibrary

        lib = FpLibrary.load('zinc_fplib.pickle')

Key checks performed
--------------------

The :class:`~chemstep.fp_library.FpLibrary` constructor will ensure:

* All filename prefixes match exactly across SMILES, ID, and fingerprint files.
* Each file trio has identical length, with a constant fingerprint length
  (``fp_len_bytes``) across the library.
* Correct ``dtype``: ``np.uint8`` for fingerprints, ``np.int64`` for IDs.

On-demand loading
-----------------

An :class:`~chemstep.fp_library.FpLibrary` stores only the file paths in memory.
Data is loaded from disk only when requested, keeping memory use minimal.
See the :class:`~chemstep.fp_library.FpLibrary` documentation for details.
