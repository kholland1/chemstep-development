===================================
Running ChemSTEP in “Manual” Mode
===================================

In *manual* mode ChemSTEP performs the *search / prioritization*
rounds, but **you** handle docking outside the loop (e.g., on a local
GPU workstation).

Round loop
----------

1. **Beacon selection**
   ``CSAlgo.get_beacons`` picks up to *max_beacons* diverse seed
   molecules from the previous round.

2. **Similarity search** (array jobs)
   ``SearchJob`` computes mint-D distances chunk-wise and writes:

   * ``mintdsXXXX.npy`` – per-molecule minimum distance
   * ``exclXXXX.npy``   – exclusion flags

3. **Pick molecules to dock**
   ``CSAlgo.get_todock_list`` collects the top
   ``n_docked_per_round`` candidates, writing
   ``smi_round_N.smi`` and ``absolute_ids_round_N.txt`` in
   *complete_info_dir*.

4. **Dock externally**
   You run your favourite docking software, producing
   a *score_dict* mapping ``absolute_id → docking_score``.

5. **Seed next round**

   .. code-block:: python

      with open('round1_scores.pkl', 'wb') as fh:
          pickle.dump(score_dict, fh)

      scores_dict = pickle.load(open('round1_scores.pkl', 'rb'))
      algo.run_one_round(round_n=2, scores_dict=scores_dict)

Cluster submission helpers
--------------------------

ChemSTEP provides two convenience wrappers for launching array jobs:

* :py:class:`chemstep.job_array.SlurmJobArray` – builds a SLURM batch script, submits it, and polls for completion.
* :py:class:`chemstep.job_array.SGEJobArray` – equivalent helper for Sun Grid Engine / UGE clusters.


These autogenerate array scripts (`run_array_round_N.sh`) with sensible
defaults that you can override via ``slurm_options``/``sge_options``.

Tips & pitfalls
---------------

* **Flush exclusions**: mint-D updates are `memmap`-backed; always wait
  for array jobs to finish (or call ``CSAlgo.all_jobs_completed``)
  before starting the next round.
* **Score orientation**: ChemSTEP assumes *lower* scores are *better*
  (typical of docking).  For scoring schemes where *higher* is better,
  simply negate.