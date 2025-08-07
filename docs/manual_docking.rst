=================================
Running ChemSTEP in “manual” mode
=================================

In *manual* mode, ChemSTEP performs the search and selection of new molecules to dock, which you then
build and dock on your own, and feed the scores back to ChemSTEP. For now, this is the best way to do tera-scale
docking since you have full control over your docking, and the time involved for the docking is already on the order of
hours/days of compute per round.

First step: seed round
----------------------

You need to provide a set of randomly selected molecules from your library, along with their docking scores, to start
the first ChemSTEP round. For the dopamine D4 example, we used ~138k molecules (.1% of the 138M library) as the seed
set, which can be found in the `d4lib/seed_dicts/seed_scores.npy` and `d4lib/seed_dicts/seed_indices.npy` files. These
files specify the docking score and absolute indices (in the FpLibrary) of the seed molecules, respectively, and are set
in the parameters file. Here's an example parameter file, which you can call `chemstep_params.txt`:

   .. code-block:: text

      seed_indices_file: /wynton/group/bks/work/omailhot/d4lib/seed_dicts/seed_indices.npy
      seed_scores_file: /wynton/group/bks/work/omailhot/d4lib/seed_dicts/seed_scores.npy
      hit_pprop: 4
      n_docked_per_round: 100000
      max_beacons: 100
      max_n_rounds: 100

From these, you can seed ChemSTEP as follows:

   .. code-block:: python

      from chemstep import CSAlgo
      from chemstep.fp_library import load_library_from_pickle

      fplib = load_library_from_pickle('/wynton/group/bks/work/omailhot/d4lib/fplib.pickle')
      algo = CSAlgo(fplib, 'chemstep_params.txt', 'output_directory', 32)

The 4 required arguments to `CSAlgo` are:

1. The fingerprint library, which you can load from a pickle file.

2. The path to the parameters file.

3. The output directory where ChemSTEP will save its results.

4. The number of parallel processes in the main job. 32 is a good number (the main job will need to run with 32 CPUs).

