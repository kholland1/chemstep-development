=============
API Reference
=============

ChemSTEP’s main user-facing classes live in :py:mod:`chemstep.algo`,
:py:mod:`chemstep.parameters` and :py:mod:`chemstep.fp_library`.

.. autofunction:: chemstep.version

.. autoclass:: chemstep.algo.CSAlgo
   :members: run_local, run_slurm_array, get_todock_list
   :undoc-members:

.. autofunction:: chemstep.parameters.read_param_file

.. autoclass:: chemstep.parameters.CSParams
   :members:

.. autofunction:: chemstep.fp_library.load_library_from_pickle

.. autoclass:: chemstep.fp_library.FpLibrary
   :members:
   :undoc-members:

.. autoclass:: chemstep.chaining_log.ChainingLog
   :members:
   :undoc-members:

.. autofunction:: chemstep.chaining_log.save_flush_np

.. autofunction:: chemstep.chaining_log.update_mintds

.. autofunction:: chemstep.chaining_log.get_mintd_distrib
