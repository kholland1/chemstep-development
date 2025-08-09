=================================
ChemSTEP parameters/configuration
=================================

This page describes the main configuration parameters that control a
ChemSTEP run. These parameters are typically stored in a plain-text
parameter file, the path to which is passed to :class:`~chemstep.algo.CSAlgo`
when creating a new ChemSTEP run.

Parameter file format
---------------------

The parameter file must contain one ``key: value`` pair per line.
Order doesn't matter. The following keys are required:

1. ``seed_indices_file``
2. ``seed_scores_file``
3. ``hit_pprop``
4. ``n_docked_per_round``
5. ``max_beacons``
6. ``max_n_rounds``

Example:

.. code-block:: text

   seed_indices_file: seeds.indices.npy
   seed_scores_file: seeds.scores.npy
   hit_pprop: 4.0
   n_docked_per_round: 100000
   max_beacons: 100
   max_n_rounds: 100

Parameter descriptions
----------------------

``seed_indices_file`` (str)
   Path to a NumPy ``.npy`` file (``int64`` dtype) containing the
   **full indices** of the initial docked set. These indices must be
   compatible with the :class:`~chemstep.fp_library.FpLibrary` used
   for the run.

``seed_scores_file`` (str)
   Path to a NumPy ``.npy`` file (``float32`` dtype) containing docking
   scores for the initial docked set, in the same order as
   ``seed_indices_file``.

``hit_pprop`` (float)
   The pProportion threshold defining a docking "hit". For example,
   ``4.0`` means the top ``10**(-4)`` fraction (0.01%) of the library.
   Used to determine the score cutoff for selecting beacons.

``n_docked_per_round`` (int)
   Number of molecules to prioritize, build, and dock in each round.

``max_beacons`` (int)
   Maximum number of beacons retained at any time. Beacons are top
   scoring molecules that guide further exploration and are selected
   to be maximally diverse from one another.

``max_n_rounds`` (int)
   Maximum number of chaining rounds to perform.

