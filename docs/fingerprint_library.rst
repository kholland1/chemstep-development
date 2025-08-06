===========================
Creating a Fingerprint Mesh
===========================

This tutorial shows how to build a **FpLibrary** object from raw
SMILES files and pre-computed Morgan fingerprints.

Prerequisites
-------------

* SMILES list split into chunks (one per future array job)
* Corresponding ZINC (or other) integer IDs
* Binary fingerprints stored as contiguous **uint8** NumPy arrays
  (shape: *n_molecules × fp_len_bytes*)

Workflow overview
-----------------

1. **Generate fingerprints** (RDKit)

   .. code-block:: python

      from rdkit import Chem
      from rdkit.Chem import AllChem
      import numpy as np

      def morgan_fp(smiles, nbits=2048, radius=2):
          mol = Chem.MolFromSmiles(smiles)
          fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
          return np.asarray(fp, dtype=np.uint8)

2. **Save data**
   Save *fpsXXXX.npy*, *zidsXXXX.npy*, and *smiXXXX.smi* where
   **XXXX** is a numeric/alpha suffix shared by the trio.

3. **Instantiate** ``FpLibrary``

   .. code-block:: python

      lib = FpLibrary(
          fingerprint_files = sorted(glob('fps*.npy')),
          id_files          = sorted(glob('zids*.npy')),
          smi_files         = sorted(glob('smi*.smi')),
          outname           = 'zinc.fp_lib.pickle'
      )

Key checks performed
--------------------

* All three filename prefixes match **exactly**.
* Each trio has identical length; *fp_len_bytes* is constant.
* ``dtype`` validation (``np.uint8`` for fps, ``np.int64`` for IDs).

Loading data on demand
----------------------

``FpLibrary`` keeps only filenames in memory.  Fingerprints/IDs are
memory-mapped when needed, enabling petabyte-scale libraries.

.. automethod:: chemstep.fp_library.FpLibrary.load_fps
   :noindex:

.. automethod:: chemstep.fp_library.FpLibrary.get_full_index
   :noindex:
