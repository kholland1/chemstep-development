==========
Quickstart
==========

.. code-block:: python

   from chemstep.fp_library import FpLibrary
   from chemstep.algo      import CSAlgo

   # 1)  Load the pre-computed fingerprint library (.pickle)
   lib = FpLibrary.load_library_from_pickle('zinc.fp_lib.pickle')

   # 2)  Parse parameters from a YAML/TXT file
   algo = CSAlgo(lib, 'params.txt', output_directory='run1',
                 n_proc=32, docking_method='manual')

   # 3)  Run a full chaining campaign (manual docking mode)
   algo.linking_loop()
