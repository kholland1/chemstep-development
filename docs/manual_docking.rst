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

      fplib = load_library_from_pickle('/wynton/group/bks/work/omailhot/d4_fplib.pickle')
      algo = CSAlgo(fplib, 'chemstep_params.txt', 'output_directory', 32, verbose=True)

The 4 required arguments to `CSAlgo` are:

1. The fingerprint library, which you can load from a pickle file.

2. The path to the parameters file.

3. The output directory where ChemSTEP will save its results.

4. The number of parallel processes in the main job. 32 is a good number (the main job will need to run with 32 CPUs).

Then, the first step of the manual docking pipeline is to run round 1 of prioritization, which selects beacons from
your already docked molecules, and also sets the score threshold according to the `hit_pprop` parameter in the
parameters file. This is done by adding the following 2 lines to the script we already had:

   .. code-block:: python

      from chemstep import CSAlgo
      from chemstep.fp_library import load_library_from_pickle

      fplib = load_library_from_pickle('/wynton/group/bks/work/omailhot/d4_fplib.pickle')
      algo = CSAlgo(fplib, 'chemstep_params.txt', 'output_directory', 32, verbose=True)

      indices, scores = algo.seed()
      algo.run_one_round(1, indices, scores)

Assuming you didn't change the default `pickle_prefix` argument in the `CSAlgo` constructor, this will save the
results of the first round in the working directory under the name `chemstep_algo_1.pickle`. You will also get
a .smi file written to the `complete_info` directory, which again, assuming you didn't change the defaults, will be
located at::

   output_directory/complete_info/smi_round_1.smi

Second step: manual docking
---------------------------

This being manual docking mode, you then proceed to dock this file as you normally would, making sure the keep the
alphanumeric molecule IDs associated with the docking scores.

The .smi file is formatted according to the standard, as follows:

   .. code-block:: text

      <SMILES> <alphanumeric ID>
      <SMILES> <alphanumeric ID>
      ...

Using the default parameters, the alphanumeric IDs will look something like this:

   .. code-block:: text

      <SMILES> CSLB00000000BR
      <SMILES> CSLB000009mkyE

You can then dock these molecules using your preferred docking software, and once you have the scores, you can
feed them back to ChemSTEP by creating a new file with the same format as the seed scores file, i.e., a
`.npy` file with the scores and indices of the docked molecules. The indices should be the absolute indices in the
fingerprint library (which can be obtained from the alphanumeric IDs, see below) and the scores should be the docking
scores you obtained.

Let's assume you have a .txt file with the docking scores, named `scores_round_1.txt` and formatted as follows:

   .. code-block:: text

      CSLB00000000BR -7.5
      CSLB000009mkyE -8.0
      ...

You can convert this to the required `.npy` format using the following script:

   .. code-block:: python

      import numpy as np
      from chemstep.id_helper import char_to_int64

      with open('scores_round_1.txt', 'r') as f:
          lines = f.readlines()

      scores = np.zeros(len(lines), dtype=np.float32)
      indices = np.zeros(len(lines), dtype=np.int64)
      for i, line in enumerate(lines):
          lib_id, score = line.strip().split()
          scores[i] = float(score)
          indices[i] = char_to_int64(lib_id)
      np.save('scores_round_1.npy', scores)
      np.save('indices_round_1.npy', indices)


Last step: iterate
------------------

You can then feed these scores back to ChemSTEP and run the second round with the following script, called with `2`
as its only command-line argument:

    .. code-block:: python

        from chemstep.algo import load_from_pickle
        import sys

        round_n = int(sys.argv[1])
        saved_algo = load_from_pickle(f'chemstep_algo_{round_n - 1}.pickle')

        indices = np.load(f'indices_round_{round_n - 1}.npy')
        scores = np.load(f'scores_round_{round_n - 1}.npy')

        algo.run_one_round(round_n, indices, scores)

You can then repeat the procedure for as many rounds as needed. The performance is reported in
`output_directory/complete_info/run_summary.df`, which contains the number of beacons selected, the
number of molecules docked, the number of hits found, the distance threshold for the selected molecules to dock,
and the last added beacon's distance to all previous beacons.