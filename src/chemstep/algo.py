import numpy as np
from chemstep.fp_library import FpLibrary
from chemstep.parameters import CSParams, read_param_file
from chemstep.chaining_log import ChainingLog
from chemstep.search_job import SearchJob
from chemstep.lookup_docking import LookupDocking
import pickle
from multiprocessing import Pool
from numba import njit
from chemstep.fingerprints import get_tanimoto_max
import os
from chemstep.job_array import SlurmJobArray, SGEJobArray, SlurmNodeArray
from chemstep.utils import read_np_data
from chemstep.id_helper import int64_to_char, char_to_int64
from numpy.lib.format import open_memmap
from chemstep.bookkeeper import Bookkeeper
import threading
import glob
import time
from chemstep.autodock_algo import AutoDocking

def load_from_pickle(fn):
    """
    Load a previously pickled :class:`CSAlgo` object from disk.

    This is the recommended way to resume an interrupted or completed
    ChemSTEP run. The returned object will contain all current state
    (parameters, chaining log, beacon history, etc.) exactly as when
    it was pickled.

    Parameters
    ----------
    fn : str
        Path to the pickle file containing the saved :class:`CSAlgo`
        instance.

    Returns
    -------
    CSAlgo
        The unpickled :class:`CSAlgo` object, ready to continue
        processing.

    Raises
    ------
    AssertionError
        If the unpickled object is not an instance of :class:`CSAlgo`.

    Notes
    -----
    * The pickle contains file paths (e.g., to NumPy `.npy` data) and
      other run-specific directories. These may be relative paths.
    * To avoid missing-file errors, load the pickle in the **same
      environment** (directory structure, Python version, and installed
      libraries) as the one in which it was created.
    * If paths were relative when pickled, you should restore them
      from the same working directory.
    """
    with open(fn, 'rb') as f:
        obj = pickle.load(f)
        assert isinstance(obj, CSAlgo)
        return obj


class CSAlgo:
    """ChemSTEP (Chemical Space Traversal and Exploration Procedure) controller that orchestrates chaining rounds.

    A ``CSAlgo`` instance owns the run state: the input library, user
    parameters, persistent logging, job orchestration (local or on a
    scheduler), and selection of maximally diverse beacons across rounds.
    It can be pickled between rounds to make resuming trivial, for instance in the case of
    manual docking where the user generates the docking scores to feed back into ChemSTEP for
    the next round of search.

    Parameters
    ----------
    fp_lib : FpLibrary
        Precomputed fingerprint/ID/SMILES library to traverse.
    chemstep_params : CSParams | str
        Either a :class:`~chemstep.parameters.CSParams` instance or a path to
        a parameter file parseable via :func:`~chemstep.parameters.read_param_file`.
    output_directory : str
        Directory where ChemSTEP writes logs, intermediate arrays, and jobs.
    n_proc : int
        Number of local worker processes when running without a scheduler.
    use_pickle : bool, default=True
        If ``True``, write a pickle of ``self`` at the end of each round.
    pickle_prefix : str, default="chemstep_algo"
        Prefix for per‑round pickles (``{prefix}_{round}.pickle``).
    verbose : bool, default=False
        If ``True``, print progress messages.
    skip_setup : bool, default=False
        If ``True``, reuse existing chaining files (no zero‑fill init).
    info_dir : str | None, default=None
        Directory for human‑readable per‑round artifacts
        (SMI, histograms, docked lists). Defaults to
        ``{output_directory}/complete_info``.
    docking_method : {"manual","lookup"}, default="manual"
        How to obtain scores for prioritized molecules. ``"manual"`` writes
        a SMI list and returns ``(None, None)``; ``"lookup"`` fetches scores
        from precomputed arrays given in ``scores_fns``.
    smi_id_prefix : str, default="CSLB"
        Prefix used when writing SMILES IDs (e.g., ``CSLB000…``).
    scheduler : {"slurm","sge",None}, default=None
        If set, distribute search jobs as an array on the chosen scheduler.
        Otherwise, run locally with multiprocessing.
    python_exec : str | None, default=None
        Python executable to use inside scheduler array tasks (e.g., the path
        to your venv’s ``python``). If ``None`` with a scheduler, defaults to
        ``"python"``.
    slurm_options : dict | None, default=None
        Extra/override ``#SBATCH`` options (e.g., ``{"time": "2:00:00"}``).
    sge_options : dict | None, default=None
        Extra/override ``#$ -`` options for SGE arrays.
    scores_fns : list[str] | None, default=None
        Required when ``docking_method="lookup"``. One path per library file
        to a ``float32`` ``.npy`` of scores aligned to IDs.

    Attributes
    ----------
    output_directory : str
        Base directory for run artifacts and chaining arrays.
    fp_lib : FpLibrary
        The working library.
    params : CSParams
        Parsed parameters for this run.
    chaining_log : ChainingLog
        Manager for exclusions, minTD arrays, and histograms.
    book : Bookkeeper
        Streams ``run_summary.df`` and ``beacons.df`` as the run progresses.
    score_thresh : float | None
        Current docking score threshold defining a “hit”.
    unused_beacons : list[tuple[float, int]]
        Candidate beacons as ``(score, full_index)`` pairs not yet used.
    current_beacons : list[tuple[float, int]]
        Beacons used in the *most recent* search round.
    current_beacons_dists : list[float]
        Per‑beacon novelty distances (1 − max Tanimoto) against the set of
        previously used beacons when they were selected.
    current_mintd_thresh : float | None
        The continuous minTD threshold (bin center) used to choose molecules
        to dock this round.
    used_beacons_fps : np.ndarray[uint8]
        Ring buffer of fingerprints for all previously used beacons with
        shape ``(max_n_rounds * max_beacons, fp_len_bytes)``.
    used_beacons_count : int
        Number of rows currently valid in ``used_beacons_fps``.

    Notes
    -----
    * “minTD” refers to *minimum Tanimoto distance* = ``1 − max_tanimoto``.
    * Exclusions are updated immediately when a molecule is selected
      to dock so it cannot be re‑selected in later rounds.

    See Also
    --------
    chemstep.parameters.CSParams
    chemstep.fp_library.FpLibrary
    chemstep.chaining_log.ChainingLog
    """
    def __init__(self, fp_lib, chemstep_params, output_directory, n_proc, use_pickle=True,
                 pickle_prefix="chemstep_algo", verbose=False, skip_setup=False, info_dir=None, docking_method="manual",
                 smi_id_prefix="CSLB", scheduler=None, python_exec=None, slurm_options=None, sge_options=None,
                 scores_fns=None, slurm_node_array=False, slurm_tasks_per_node=64, use_logfile=True, max_reruns=5, dockfiles_path=None):
        os.makedirs(output_directory, exist_ok=True)
        self.output_directory = output_directory
        if use_pickle:
            assert pickle_prefix is not None
        self.use_pickle = use_pickle
        self.pickle_prefix = pickle_prefix
        assert isinstance(fp_lib, FpLibrary)
        self.fp_lib = fp_lib
        if isinstance(chemstep_params, CSParams):
            self.params = chemstep_params
        elif isinstance(chemstep_params, str):
            self.params = read_param_file(chemstep_params)
        else:
            raise ValueError("chemstep_params has to be a string path to a parameter file or an instance of CSParams")
        self.scores_fns = scores_fns
        self.n_proc = n_proc
        self.verbose = verbose
        if info_dir is None:
            info_dir = os.path.join(output_directory, "complete_info")
        os.makedirs(info_dir, exist_ok=True)
        self.info_dir = info_dir
        self.docking_method = docking_method
        self.smi_id_prefix = smi_id_prefix
        self.book = Bookkeeper(self.info_dir, self.smi_id_prefix)
        if scheduler is not None:
            if scheduler not in ["slurm", "sge"]:
                raise ValueError("Scheduler must be either 'slurm' or 'sge'")
            if python_exec is None:
                python_exec = "python"
                print("WARNING: no python_exec specified, using 'python' by default. (on some schedulers this will" +
                      "lead to using the system Python, which may not have all dependencies installed)")
        self.python_exec = python_exec
        self.scheduler = scheduler
        self.slurm_node_array = bool(slurm_node_array)
        self.slurm_tasks_per_node = int(slurm_tasks_per_node)
        self.use_logfile = use_logfile
        self.logfile = None
        self.max_reruns = int(max_reruns)
        if self.use_logfile:
            self.logfile = f"{self.pickle_prefix}.log"
            with open(self.logfile, 'a') as f:
                f.write(f"=== New ChemSTEP log started at {time.asctime()} ===\n")
        if slurm_options is None:
            slurm_options = {
                "account": "rrg-mailhoto",
                "ntasks": "1",
                "mem": "4GB",
                "nodes": "1",
                "cpus-per-task": "1",
                "time": "1:00:00"
            }
        self.slurm_options = slurm_options
        if sge_options is None:
            sge_options = {
                "l": "h_rt=12:00:00",
                "S": "/bin/bash",
                "P": "shoichetlab"
            }
        self.sge_options = sge_options
        self.print_verbose("about to setup ChainingLog")
        if skip_setup:
            self.chaining_log = ChainingLog(self.fp_lib, output_directory, self.n_proc, write_empty_files=False)
        else:
            self.chaining_log = ChainingLog(self.fp_lib, output_directory, self.n_proc)
        self.print_verbose("ChainingLog set")
        self.score_thresh = None
        self.unused_beacons = []
        self.current_beacons = []
        self.current_beacons_dists = []
        self.current_mintd_thresh = None
        self.used_beacons_fps = np.zeros((self.params.max_n_rounds * self.params.max_beacons,
                                          self.fp_lib.fp_length_bytes), dtype=np.uint8)
        self.used_beacons_count = 0
        self.dockfiles_path = dockfiles_path

    def print_verbose(self, s):
        """Print a message only if ``verbose`` is enabled.

        Parameters
        ----------
        s : str
            Message to print.
        """
        if self.verbose:
            if self.use_logfile:
                with open(self.logfile, 'a') as f:
                    f.write(s + '\n')
            print(s)

    def run_one_round(self, round_n, new_indices, new_scores):
        """Execute a complete ChemSTEP round.

        Steps:
          1) Log previous round summary (if ``round_n > 1``).
          2) Select maximally beacons from all found hits still unused as beacons (includes latest docked hits).
          3) Search the whole library for closest neighbors (update minTD).
          4) Choose molecules to dock based on global minTD histogram.
          5) Produce SMILES/ID lists and either return ``None`` (manual) or
             lookup new scores and return them.

        Parameters
        ----------
        round_n : int
            1‑based round index.
        new_indices : np.ndarray | None
            Full indices from the **previous** docking batch (or ``None`` for
            the first call when using ``manual`` docking).
        new_scores : np.ndarray | None
            Scores aligned to ``new_indices`` (or ``None`` for the first call
            when using ``manual`` docking).

        Returns
        -------
        (np.ndarray | None, np.ndarray | None)
            ``(new_indices, new_scores)`` for this round when using
            ``docking_method="lookup"``; ``(None, None)`` for ``"manual"``.

        Raises
        ------
        RuntimeError
            If array jobs did not complete successfully.
        """
        if round_n > 1:
            n_hits = int(np.sum(new_scores <= self.score_thresh))
            beacon_ids = [b[1] for b in self.current_beacons]
            beacon_scores = [b[0] for b in self.current_beacons]
            self.book.log_round(round_n-1, len(new_indices), n_hits, self.current_mintd_thresh,
                                beacon_ids, beacon_scores, self.current_beacons_dists)
        beacons = self.get_beacons(new_indices, new_scores)
        t0 = time.time()
        if len(beacons) != 0:
            self.print_verbose(f"Starting round {round_n} with {len(beacons)} beacons at time {time.asctime()}")
            jobs = []
            self.is_restartable = True
            for j in range(self.fp_lib.n_files):
                unique_id = "{}_{}".format(round_n, j)
                job = SearchJob(unique_id, beacons, round_n, self.fp_lib, self.chaining_log, j, scheduler=self.scheduler)
                jobs.append(job)
            array_jobid = None
            if self.scheduler == "slurm":
                array_jobid = self.run_slurm_array(jobs, round_n)
            elif self.scheduler == "sge":
                array_jobid = self.run_sge_array(jobs, round_n)
            else:
                if self.scheduler is not None:
                    raise ValueError(f"Unsupported scheduler: {self.scheduler}")
                self.run_local(jobs)
            if self.scheduler is not None:
                if not self.all_jobs_completed(round_n):
                    raise RuntimeError(f"Not all jobs for round {round_n} completed. Check the job folder: {self.chaining_log.jobs_folder}")

            # async deletion of job pickles and outputs if everything ran smoothly
            if array_jobid is not None:
                threading.Thread(
                    target=self._cleanup_round_artifacts,
                    args=(round_n, self.scheduler, array_jobid),
                    daemon=True
                ).start()
        else:
            self.print_verbose(f"No new beacons found in round {round_n}, skipping minTD updates but will still dock")

        self.print_verbose("Starting docking for round {}".format(round_n))
        lib_array_indices, smi_list, absolute_ids = self.get_todock_list(round_n)
        self.used_beacons_fps[self.used_beacons_count:self.used_beacons_count+len(beacons)] = beacons
        self.used_beacons_count += len(beacons)

        if self.docking_method == "lookup":
            self.write_smi_file(smi_list, lib_array_indices, round_n, absolute_ids)
            new_indices, new_scores = self.lookup_dock(lib_array_indices, smi_list, round_n)
        elif self.docking_method == "manual":
            self.write_smi_file(smi_list, lib_array_indices, round_n, absolute_ids)
            new_indices, new_scores = None, None
        elif self.docking_method == "auto":
            self.write_smi_file(smi_list, lib_array_indices, round_n, absolute_ids)
            smi_file_path = "{}/smi_round_{}.smi".format(self.info_dir, round_n)
            new_indices, new_scores = self.auto_dock(lib_array_indices, smi_list, round_n, smi_file_path)
        else:
            raise ValueError(f"Docking method {self.docking_method} not yet implemented")

        if self.use_pickle:
            with open(f'{self.pickle_prefix}_{round_n}.pickle', 'wb') as f:
                pickle.dump(self, f)
        self.print_verbose(f"Round {round_n} complete at time {time.asctime()} (took {time.time() - t0:.1f} s)")
        return new_indices, new_scores

    def write_smi_file(self, smi_list, lib_array_indices, round_n, absolute_ids):
        """Write the SMI file and an accompanying absolute‑ID list for a round.

        Creates two files in ``info_dir``:
          * ``smi_round_{round_n}.smi`` containing ``<SMILES> <prefixed-ID>``.
          * ``absolute_ids_round_{round_n}.txt`` containing raw int64 IDs.

        Parameters
        ----------
        smi_list : list[str]
            SMILES selected to dock.
        lib_array_indices : np.ndarray
            Array of shape ``(N, 2)`` with ``(lib_index, array_index)``.
        round_n : int
            Current round number.
        absolute_ids : np.ndarray
            Aligned absolute IDs (typically ZINC ints).
        """
        abs_out = open(f'{self.info_dir}/absolute_ids_round_{round_n}.txt', 'w')
        with open("{}/smi_round_{}.smi".format(self.info_dir, round_n), 'w') as f:
            for smi, lib_arr, abs_id in zip(smi_list, lib_array_indices, absolute_ids):
                full_index = self.fp_lib.get_full_index(lib_arr[0], lib_arr[1])
                char_name = int64_to_char(full_index, prefix=self.smi_id_prefix)
                f.write(f'{smi} {char_name}\n')
                abs_out.write(f'{abs_id}\n')
        abs_out.close()

    def seed(self, score_thresh=None):
        """Initialize from the seed set and compute the hit threshold.

        Reads the seed indices and scores, computes the default score
        threshold from ``hit_pprop`` (unless overridden), and marks all
        seed molecules as excluded.

        Parameters
        ----------
        score_thresh : float | None, default=None
            If provided, override the automatically computed threshold.

        Returns
        -------
        (np.ndarray, np.ndarray)
            ``(seed_indices, seed_scores)`` as loaded from files.
        """
        seed_indices = np.load(self.params.seed_indices_file)
        seed_scores = np.load(self.params.seed_scores_file)
        if score_thresh is None:
            self.set_score_thresh(seed_indices, seed_scores)
            self.print_verbose(f"Automatically set score threshold to {self.score_thresh:.2f} " +
                               f"(pProp of {self.params.hit_pprop})")
        else:
            self.set_score_thresh(seed_indices, seed_scores)  # still important, to remove already docked compounds
            self.print_verbose(f"Automatically set score threshold to {self.score_thresh:.2f} " +
                               f"(pProp of {self.params.hit_pprop})")
            self.score_thresh = score_thresh
            self.print_verbose(f"Overrode score threshold to {self.score_thresh:.2f}")
        return seed_indices, seed_scores

    def linking_loop(self, score_thresh=None):
        """Run the full multi‑round chaining process. For now, this only works in 'lookup' docking
        mode, until a pure Python docking implementation is available.

        Parameters
        ----------
        score_thresh : float | None, default=None
            Optional override of the seed‑derived score threshold.
        """
        new_indices, new_scores = self.seed(score_thresh=score_thresh)
        for i in range(1, self.params.max_n_rounds+1):
            new_indices, new_scores = self.run_one_round(i, new_indices, new_scores)

    def run_local(self, jobs):
        """Execute search jobs locally with multiprocessing.

        Parameters
        ----------
        jobs : list[SearchJob]
            One job per library shard/file.
        """
        p = Pool(self.n_proc)
        p.map(_run_one_job_local, jobs)

    def run_slurm_array(self, jobs, round_n):
        """Submit search jobs as a Slurm array and wait for completion.

        Parameters
        ----------
        jobs : list[SearchJob]
            Jobs to pickle and submit.
        round_n : int
            Current round number (used in filenames).
        """
        if self.slurm_node_array:
            self.dump_job_pickles(jobs, round_n)
            launcher = SlurmNodeArray(self.chaining_log.jobs_folder, round_n, len(jobs),
                                      tasks_per_node=self.slurm_tasks_per_node, python_exec=self.python_exec,
                                      slurm_options=self.slurm_options)
            job_id = launcher.submit()
            try:
                launcher.wait()
            except Exception as e:
                self.print_verbose(f"First wait() pass failed with exception {e}")
            ok, extra_ids = self._wait_and_resubmit_incomplete(round_n, scheduler="slurm")
            if not ok:
                raise RuntimeError(f"After retries, some SearchJobs are still incomplete for round {round_n}")
            # Cleanup only when truly done
            threading.Thread(target=self._cleanup_round_artifacts, args=(round_n, self.scheduler), daemon=True).start()
            return job_id
        else:
            self.dump_job_pickles(jobs, round_n)
            job_array = SlurmJobArray(round_n=round_n, n_jobs=len(jobs),
                                      job_folder=self.chaining_log.jobs_folder,
                                      python_exec=self.python_exec, slurm_options=self.slurm_options)
            job_id = job_array.submit()
            try:
                job_array.wait(job_id)
            except Exception as e:
                self.print_verbose(f"First wait() pass failed with exception {e}")
            ok, extra_ids = self._wait_and_resubmit_incomplete(round_n, scheduler="slurm")
            if not ok:
                raise RuntimeError(f"After retries, some SearchJobs are still incomplete for round {round_n}")
            threading.Thread(target=self._cleanup_round_artifacts, args=(round_n, self.scheduler), daemon=True).start()
            return job_id

    def run_sge_array(self, jobs, round_n):
        """Submit search jobs as an SGE array and wait for completion.

        Parameters
        ----------
        jobs : list[SearchJob]
            Jobs to pickle and submit.
        round_n : int
            Current round number (used in filenames).
        """
        self.dump_job_pickles(jobs, round_n)
        job_array = SGEJobArray(round_n=round_n, n_jobs=len(jobs),
                                job_folder=self.chaining_log.jobs_folder,
                                python_exec=self.python_exec, sge_options=self.sge_options)
        job_id = job_array.submit()
        try:
            job_array.wait(job_id)
        except Exception as e:
            self.print_verbose(f"First wait() pass failed with exception {e}")
        ok, extra_ids = self._wait_and_resubmit_incomplete(round_n, scheduler="sge")
        if not ok:
            raise RuntimeError(f"After retries, some SearchJobs are still incomplete for round {round_n}")
        threading.Thread(target=self._cleanup_round_artifacts, args=(round_n, self.scheduler), daemon=True).start()
        return job_id

    def dump_job_pickles(self, jobs, round_n):
        """Serialize per‑file :class:`SearchJob` objects to the jobs folder.

        Filenames follow ``{round_n}_{file_index}.pickle``.

        Parameters
        ----------
        jobs : list[SearchJob]
            Jobs to serialize.
        round_n : int
            Round used in the filename stem.
        """
        for j, job in enumerate(jobs):
            with open(os.path.join(self.chaining_log.jobs_folder,
                                   f"{round_n}_{j}.pickle"), "wb") as f:
                pickle.dump(job, f)

    def get_todock_list(self, round_n):
        """Choose the next batch to dock based on global minTD.

        Computes the global minTD histogram across all library shards,
        selects the smallest minTD bin whose cumulative count meets
        ``n_docked_per_round``, and returns all still‑eligible molecules
        at or below that bin. Eligible molecules are marked excluded
        immediately to prevent reselection.

        Parameters
        ----------
        round_n : int
            Current round (used only for logging filenames).

        Returns
        -------
        (np.ndarray, list[str], np.ndarray)
            ``(lib_array_indices, smiles, absolute_ids)`` where
            ``lib_array_indices`` has shape ``(N, 2)``.

        Notes
        -----
        Also writes ``mintd_distrib_{round_n}.df`` for inspection.
        """
        mintd_distrib = self.chaining_log.load_global_mintd_distrib()
        with open(f"{self.info_dir}/mintd_distrib_{round_n}.df", 'w') as f:
            f.write("mintd count\n")
            for mintd_bin, n in enumerate(mintd_distrib):
                f.write(f"{(mintd_bin + 1) / 1000} {n}\n")

        target_n = self.params.n_docked_per_round
        assert np.sum(mintd_distrib) > target_n

        mintd_bin_thresh = None
        count = 0
        for b in range(1001):
            count += mintd_distrib[b]
            if count >= target_n:
                mintd_bin_thresh = b
                break
        assert mintd_bin_thresh is not None
        self.current_mintd_thresh = (mintd_bin_thresh + 0.5) / 1000
        self.print_verbose(f"minTD threshold for round {round_n} set to {self.current_mintd_thresh:.3f}")

        args = [(i, mintd_bin_thresh, self.fp_lib, self.chaining_log) for i in range(self.fp_lib.n_files)]

        with Pool(self.n_proc) as pool:
            results = pool.starmap(_process_single_lib_chunked, args)

        all_lib_array_indices = []
        all_smiles = []
        all_absolute_ids = []

        for lib_array_inds, smiles, abs_ids in results:
            all_lib_array_indices.append(lib_array_inds)
            all_smiles.extend(smiles)
            all_absolute_ids.append(abs_ids)

        lib_array_indices = np.concatenate(all_lib_array_indices, axis=0)
        absolute_ids = np.concatenate(all_absolute_ids, axis=0)

        return lib_array_indices, all_smiles, absolute_ids
        
    def auto_dock(self, lib_array_indices, smi_list, round_n, smi_file_path):
        """Retrieve scores for a batch using precomputed arrays.

        Parameters
        ----------
        lib_array_indices : np.ndarray
            Array of ``(lib_index, array_index)`` pairs to score.
        smi_list : list[str]
            SMILES (unused for lookup; kept for interface symmetry).
        round_n : int
            Round number for output filenames.

        Returns
        -------
        (np.ndarray, np.ndarray)
            ``(full_indices, scores)`` aligned arrays for the batch.

        Raises
        ------
        AssertionError
            If ``scores_fns`` is missing or misaligned with the library.
        """
        self.print_verbose("About to start building")
        docker = AutoDocking(lib_array_indices, smi_list, self.fp_lib, smi_file_path, round_n, self.dockfiles_path, verbose=self.verbose)
        docker.build_all()
        self.print_verbose("Building Done")
        self.print_verbose("Starting Docking")
        new_indices = docker.dock_all()
        self.print_verbose("'Docking' done")
        new_scores = docker.get_scores_list()
        # new_indices = np.zeros(len(new_scores), dtype=np.int64)

        # for i, ((lib_index, arr_index), score) in enumerate(zip(lib_array_indices, new_scores)):
        #     full_id = self.fp_lib.get_full_index(lib_index, arr_index)
        #     new_indices[i] = full_id
        with open(f"{self.info_dir}/docked_round_{round_n}.df", 'w') as f:
            f.write("full_id score\n")
            for full_id, score in zip(new_indices, new_scores):
                f.write(f"{full_id} {score}\n")
        return new_indices, new_scores

    def lookup_dock(self, lib_array_indices, smi_list, round_n):
        """Retrieve scores for a batch using precomputed arrays.

        Parameters
        ----------
        lib_array_indices : np.ndarray
            Array of ``(lib_index, array_index)`` pairs to score.
        smi_list : list[str]
            SMILES (unused for lookup; kept for interface symmetry).
        round_n : int
            Round number for output filenames.

        Returns
        -------
        (np.ndarray, np.ndarray)
            ``(full_indices, scores)`` aligned arrays for the batch.

        Raises
        ------
        AssertionError
            If ``scores_fns`` is missing or misaligned with the library.
        """
        self.print_verbose("About to start 'docking'")
        docker = LookupDocking(lib_array_indices, smi_list, self.scores_fns, self.fp_lib, verbose=self.verbose)
        docker.dock_all()
        self.print_verbose("'Docking' done")
        new_scores = docker.get_scores_list()
        new_indices = np.zeros(len(new_scores), dtype=np.int64)

        for i, ((lib_index, arr_index), score) in enumerate(zip(lib_array_indices, new_scores)):
            full_id = self.fp_lib.get_full_index(lib_index, arr_index)
            new_indices[i] = full_id
        with open(f"{self.info_dir}/docked_round_{round_n}.df", 'w') as f:
            f.write("full_id score\n")
            for full_id, score in zip(new_indices, new_scores):
                f.write(f"{full_id} {score}\n")
        return new_indices, new_scores

    def set_score_thresh(self, seed_indices, seed_scores):
        """Compute and store the docking score threshold; exclude seeds.

        The threshold is the quantile ``10^(−hit_pprop)`` over
        ``seed_scores``. All seed molecules are marked as excluded in
        their corresponding library shards.

        Parameters
        ----------
        seed_indices : np.ndarray
            Full indices of seed molecules (``FpLibrary`` indexing).
        seed_scores : np.ndarray
            Docking scores aligned to ``seed_indices``.
        """
        lib_arr_indices = np.zeros((len(seed_scores), 2), dtype=np.int64)
        for i, seed_index in enumerate(seed_indices):
            lib_arr_indices[i] = self.fp_lib.get_lib_array_indices(seed_index)
        self.score_thresh = np.quantile(seed_scores, 10**(-1 * self.params.hit_pprop))

        lib_arr_dict = dict()
        for lib_i, arr_i in lib_arr_indices:
            if lib_i not in lib_arr_dict:
                lib_arr_dict[lib_i] = set()
            lib_arr_dict[lib_i].add(arr_i)
        args = []
        for lib_i in lib_arr_dict:
            args.append((self.chaining_log,
                         lib_i,
                         np.array(list(lib_arr_dict[lib_i]), dtype=np.int64),
                         self.fp_lib.lengths[lib_i]))
        with Pool(self.n_proc) as p:
            p.starmap(_add_exclusions_one_index, args)

    def get_beacons(self, new_indices, new_scores):
        """Select a maximally diverse set of beacons from latest hits (max-min Tanimoto distance to previous beacons).

        Parameters
        ----------
        new_indices : np.ndarray
            Full indices for the newly docked molecules.
        new_scores : np.ndarray
            Corresponding docking scores.

        Returns
        -------
        np.ndarray
            Fingerprints with shape ``(≤ max_beacons, fp_len_bytes)`` for
            the beacons to use in the next search round.
        """

        beacons_candidates = [
            (score, idx)
            for idx, score in zip(new_indices, new_scores)
            if score <= self.score_thresh
        ]
        self.unused_beacons.extend(beacons_candidates)
        self.unused_beacons.sort()  # minimum score is best

        selected_fps = self.apply_beacons_diversity()

        return selected_fps

    def screen_novelty(self):
        raise ValueError("screen_novelty() not yet implemented")

    def apply_beacons_diversity(self):
        """Apply the configured beacon diversity strategy.

        Uses the strategy specified in params.beacon_diversity_strategy:
        - "maxdiv": Greedy max-diversity using Tanimoto distance (default)
        - "entropy_bits": Maximize fingerprint bit entropy
        - "mutual_info": Minimize mutual information between selected molecules

        Returns
        -------
        np.ndarray
            Fingerprints for the selected beacons.

        Raises
        ------
        ValueError
            If beacon_diversity_strategy is not recognized.
        """
        strategy = getattr(self.params, 'beacon_diversity_strategy', 'maxdiv')

        if strategy == "maxdiv":
            return self.apply_beacons_diversity_maxdiv()
        elif strategy == "entropy_bits":
            return self.apply_beacons_diversity_entropy_bits()
        elif strategy == "mutual_info":
            return self.apply_beacons_diversity_mutual_info()
        else:
            raise ValueError(f"Unknown beacon diversity strategy: {strategy}. " +
                           "Valid options are: 'maxdiv', 'entropy_bits', 'mutual_info'")

    def load_all_fps_unused_beacons(self):
        """Load fingerprints for all currently unused beacon candidates.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_unused, fp_len_bytes)``.
        """
        all_fps = np.zeros((len(self.unused_beacons), self.fp_lib.fp_length_bytes), dtype=np.uint8)

        # dict of dicts to untangle library indices
        lib_arr_dict = dict()

        for i, (_, full_index) in enumerate(self.unused_beacons):
            lib_index, arr_index = self.fp_lib.get_lib_array_indices(full_index)
            if lib_index not in lib_arr_dict:
                lib_arr_dict[lib_index] = dict()
            lib_arr_dict[lib_index][arr_index] = i

        for lib_index in lib_arr_dict:
            sub_indices = []
            for arr_index in lib_arr_dict[lib_index]:
                sub_indices.append(arr_index)
            fps = self.fp_lib.load_fps_subset(lib_index, sub_indices)
            for arr_index, fp in zip(sub_indices, fps):
                all_fps[lib_arr_dict[lib_index][arr_index]] = fp
        return all_fps

    def apply_beacons_diversity_maxdiv(self):
        """Greedy max‑diversity (farthest‑first) selection of beacons.

        Iteratively picks the candidate with maximum minTD to the set of
        already used beacons; updates the running distance vector to
        enforce diversity, and keeps at most ``max_beacons``.

        Returns
        -------
        np.ndarray
            Fingerprints of the selected beacons (rows correspond to the
            chosen candidates). Also updates ``current_beacons``,
            ``current_beacons_dists``, and prunes ``unused_beacons`` in‑place.
        """
        all_fps = self.load_all_fps_unused_beacons()
        if self.used_beacons_count > 0:
            distance_vector = 1 - get_tanimoto_max(self.used_beacons_fps[:self.used_beacons_count], all_fps)
        else:
            distance_vector = np.ones(len(all_fps))
        selected = np.zeros(len(all_fps), dtype=np.uint8)
        kept_beacons = []
        distances = []
        while len(kept_beacons) < self.params.max_beacons and np.sum(selected) < len(selected):
            max_index = np.argmax(distance_vector)
            distances.append(distance_vector[max_index])
            selected[max_index] = 1
            kept_beacons.append(self.unused_beacons[max_index])
            distance_vector = np.minimum(distance_vector, 1 - get_tanimoto_max(np.array([all_fps[max_index]]), all_fps))
            self.print_verbose(f"Selected beacon {self.unused_beacons[max_index][1]} with score {self.unused_beacons[max_index][0]:.2f} and maxminTD {distances[-1]:.3f}")
        self.unused_beacons = [x for i, x in enumerate(self.unused_beacons) if selected[i] == 0]
        self.current_beacons = kept_beacons
        self.current_beacons_dists = distances
        return all_fps[selected == 1]

    def apply_beacons_diversity_entropy_bits(self):
        """Select beacons that maximize entropy of fingerprint bit patterns.

        Greedily selects molecules to maximize the Shannon entropy of
        fingerprint bit positions across the selected set.

        Returns
        -------
        np.ndarray
            Fingerprints of the selected beacons (rows correspond to the
            chosen candidates). Also updates ``current_beacons``,
            ``current_beacons_dists``, and prunes ``unused_beacons`` in‑place.
        """
        all_fps = self.load_all_fps_unused_beacons()
        if len(all_fps) == 0:
            self.current_beacons = []
            self.current_beacons_dists = []
            return np.zeros((0, self.fp_lib.fp_length_bytes), dtype=np.uint8)

        def compute_bit_entropy(fps_subset):
            """Compute Shannon entropy across fingerprint bit positions."""
            if len(fps_subset) == 0:
                return 0.0
            # Convert to bit matrix (n_molecules x n_bits)
            bit_matrix = np.unpackbits(fps_subset, axis=1)
            # Compute probability of bit=1 for each position
            bit_probs = np.mean(bit_matrix, axis=0)
            # Avoid log(0) by clipping
            bit_probs = np.clip(bit_probs, 1e-10, 1-1e-10)
            # Shannon entropy for each bit position
            entropies = -bit_probs * np.log2(bit_probs) - (1-bit_probs) * np.log2(1-bit_probs)
            return np.sum(entropies)

        # Greedy selection maximizing total bit entropy
        selected_indices = []
        entropies = []

        for _ in range(min(self.params.max_beacons, len(all_fps))):
            best_entropy = -1
            best_idx = -1

            for i in range(len(all_fps)):
                if i in selected_indices:
                    continue
                # Test entropy of current selection + candidate
                candidate_fps = all_fps[selected_indices + [i]]
                entropy = compute_bit_entropy(candidate_fps)
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_idx = i

            if best_idx >= 0:
                selected_indices.append(best_idx)
                entropies.append(best_entropy)
                self.print_verbose(f"Selected beacon {self.unused_beacons[best_idx][1]} with score {self.unused_beacons[best_idx][0]:.2f} and entropy {best_entropy:.3f}")

        # Update instance variables
        selected = np.zeros(len(all_fps), dtype=np.uint8)
        selected[selected_indices] = 1
        kept_beacons = [self.unused_beacons[i] for i in selected_indices]
        self.unused_beacons = [x for i, x in enumerate(self.unused_beacons) if selected[i] == 0]
        self.current_beacons = kept_beacons
        self.current_beacons_dists = entropies

        return all_fps[selected == 1]

    def apply_beacons_diversity_mutual_info(self):
        """Select beacons minimizing mutual information between selected molecules.

        Greedily selects molecules to minimize pairwise mutual information,
        reducing redundancy between selected beacons.

        Returns
        -------
        np.ndarray
            Fingerprints of the selected beacons (rows correspond to the
            chosen candidates). Also updates ``current_beacons``,
            ``current_beacons_dists``, and prunes ``unused_beacons`` in‑place.
        """
        all_fps = self.load_all_fps_unused_beacons()
        if len(all_fps) == 0:
            self.current_beacons = []
            self.current_beacons_dists = []
            return np.zeros((0, self.fp_lib.fp_length_bytes), dtype=np.uint8)

        def mutual_information(fp1, fp2):
            """Compute mutual information between two fingerprint bit vectors."""
            # Convert to bit arrays
            bits1 = np.unpackbits(fp1)
            bits2 = np.unpackbits(fp2)

            # Joint probabilities
            p_00 = np.mean((bits1 == 0) & (bits2 == 0))
            p_01 = np.mean((bits1 == 0) & (bits2 == 1))
            p_10 = np.mean((bits1 == 1) & (bits2 == 0))
            p_11 = np.mean((bits1 == 1) & (bits2 == 1))

            # Marginal probabilities
            p_1_x = p_10 + p_11
            p_0_x = 1 - p_1_x
            p_1_y = p_01 + p_11
            p_0_y = 1 - p_1_y

            # Mutual information calculation
            mi = 0.0
            for p_xy, p_x, p_y in [(p_00, p_0_x, p_0_y), (p_01, p_0_x, p_1_y),
                                   (p_10, p_1_x, p_0_y), (p_11, p_1_x, p_1_y)]:
                if p_xy > 1e-10 and p_x > 1e-10 and p_y > 1e-10:
                    mi += p_xy * np.log2(p_xy / (p_x * p_y))

            return max(0.0, mi)  # Ensure non-negative

        # Start with the molecule that has the best score (first in sorted list)
        selected_indices = [0] if len(all_fps) > 0 else []
        mi_sums = [0.0] if len(all_fps) > 0 else []

        # Select remaining molecules minimizing total mutual information
        for _ in range(1, min(self.params.max_beacons, len(all_fps))):
            min_mi_sum = float('inf')
            best_idx = -1

            for i in range(len(all_fps)):
                if i in selected_indices:
                    continue

                # Compute total MI with already selected molecules
                mi_sum = sum(mutual_information(all_fps[i], all_fps[j])
                           for j in selected_indices)

                if mi_sum < min_mi_sum:
                    min_mi_sum = mi_sum
                    best_idx = i

            if best_idx >= 0:
                selected_indices.append(best_idx)
                mi_sums.append(min_mi_sum)
                self.print_verbose(f"Selected beacon {self.unused_beacons[best_idx][1]} with score {self.unused_beacons[best_idx][0]:.2f} and MI sum {min_mi_sum:.3f}")

        # Update instance variables
        selected = np.zeros(len(all_fps), dtype=np.uint8)
        selected[selected_indices] = 1
        kept_beacons = [self.unused_beacons[i] for i in selected_indices]
        self.unused_beacons = [x for i, x in enumerate(self.unused_beacons) if selected[i] == 0]
        self.current_beacons = kept_beacons
        self.current_beacons_dists = mi_sums

        return all_fps[selected == 1]

    def all_jobs_completed(self, round_n):
        """Check that every per‑file search job pickle is present and completed.

        Parameters
        ----------
        round_n : int
            Round to check.

        Returns
        -------
        bool
            ``True`` if all jobs report ``completed=True``, else ``False``.
        """
        for j in range(self.fp_lib.n_files):
            pickle_path = os.path.join(self.chaining_log.jobs_folder, f"{round_n}_{j}.pickle")
            if not os.path.exists(pickle_path):
                return False
            with open(pickle_path, 'rb') as f:
                job = pickle.load(f)
            if not job.completed:
                return False
        return True

    def incomplete_job_indices(self, round_n):
        """Return zero-based file indices j for which {round_n}_{j}.pickle exists and job.completed==False."""
        missing = []
        for j in range(self.fp_lib.n_files):
            p = os.path.join(self.chaining_log.jobs_folder, f"{round_n}_{j}.pickle")
            if not os.path.exists(p):
                # If a pickle is missing entirely, treat as incomplete so we re-run it.
                missing.append(j)
                continue
            with open(p, "rb") as fh:
                job = pickle.load(fh)
            if not getattr(job, "completed", False):
                missing.append(j)
        return missing

    def _submit_subset_array(self, round_n, idxs, scheduler):
        """Submit a Slurm/SGE array for just the provided zero-based idxs and wait."""
        if scheduler == "slurm":
            # Slurm accepts 0-based arrays; we pass --array=<list>
            arr_spec = _compress_index_list(idxs, one_indexed=False)
            opts = dict(self.slurm_options)
            opts["array"] = arr_spec  # override the default range
            ja = SlurmJobArray(round_n, n_jobs=len(idxs), job_folder=self.chaining_log.jobs_folder,
                               python_exec=self.python_exec, slurm_options=opts)
            job_id = ja.submit()
            try:
                ja.wait(job_id)
            except Exception as e:
                # We don't crash here; we’ll re-check incompletes and possibly retry.
                self.print_verbose(f"[resubmit error] {e}")
            return job_id

        elif scheduler == "sge":
            # SGE is 1-indexed; job script subtracts 1 internally to form pickle name
            task_spec = _compress_index_list(idxs, one_indexed=True)
            opts = dict(self.sge_options)
            opts["t"] = task_spec
            ja = SGEJobArray(round_n, n_jobs=len(idxs), job_folder=self.chaining_log.jobs_folder,
                             python_exec=self.python_exec, sge_options=opts)
            job_id = ja.submit()
            try:
                ja.wait(job_id)
            except Exception as e:
                self.print_verbose(f"[resubmit error] {e}")
            return job_id
        else:
            raise ValueError(f"Unsupported scheduler for subset submission: {scheduler!r}")

    def _wait_and_resubmit_incomplete(self, round_n, scheduler):
        """After an array (or nodearray) ends, resubmit only incomplete jobs up to max_reruns."""
        job_ids = []
        attempt = 0
        while True:
            remain = self.incomplete_job_indices(round_n)
            if not remain:
                return True, job_ids
            attempt += 1
            if attempt > self.max_reruns:
                return False, job_ids
            self.print_verbose(
                f"[retry {attempt}/{self.max_reruns}] Resubmitting {len(remain)} incomplete SearchJobs for round {round_n}")
            job_ids.append(self._submit_subset_array(round_n, remain, scheduler))

    def _cleanup_round_artifacts(self, round_n, scheduler):
        """Best-effort deletion of per-round artifacts (pickles + scheduler logs)."""
        if scheduler is None:
            return
        time.sleep(10)
        jf = self.chaining_log.jobs_folder
        try:
            # 1) All pickles for this round
            pickle_patterns = {
                os.path.join(jf, f"{round_n}_*.pickle"),
                os.path.join(jf, f"{round_n}_node_*.pickle"),
            }
            to_delete = set()
            for patt in pickle_patterns:
                to_delete.update(glob.glob(patt))
            for p in to_delete:
                if os.path.isfile(p):
                    os.remove(p)

            # 2) Scheduler logs — delete all for this round, across all job IDs
            if scheduler == "slurm":
                for patt in [
                    os.path.join(jf, f"slurm_{round_n}-*.out"),
                    os.path.join(jf, f"slurm_{round_n}-*.err"),
                    os.path.join(jf, f"slurm-node-*.out"),
                    os.path.join(jf, f"slurm-node-*.err"),
                ]:
                    for p in glob.glob(patt):
                        if os.path.isfile(p):
                            os.remove(p)
            elif scheduler == "sge":
                for patt in [
                    os.path.join(jf, f"sge_{round_n}_*.out"),
                    os.path.join(jf, f"sge_{round_n}_*.err"),
                ]:
                    for p in glob.glob(patt):
                        if os.path.isfile(p):
                            os.remove(p)
        except Exception as e:
            self.print_verbose(f"[failed cleanup r{round_n}] {e}")


def _run_one_job_local(job):
    job.run_local()


def _compress_index_list(idxs, one_indexed=False):
    """Turn [0,1,2,5,7,8] into '0-2,5,7-8' (or 1-indexed for SGE)."""
    if not idxs:
        return ""
    xs = sorted(int(i) + (1 if one_indexed else 0) for i in set(idxs))
    ranges = []
    start = prev = xs[0]
    for v in xs[1:]:
        if v == prev + 1:
            prev = v
            continue
        ranges.append((start, prev))
        start = prev = v
    ranges.append((start, prev))
    parts = [f"{a}-{b}" if a != b else f"{a}" for a, b in ranges]
    return ",".join(parts)


def _process_single_lib_chunked(lib_index, bin_thresh, fp_lib, chaining_log, chunk_size=100_000):
    mintd_path = chaining_log.get_filename(chaining_log.mintd_prefix, chaining_log.get_suffix(lib_index))
    excl_path = chaining_log.get_filename(chaining_log.exclusion_prefix, chaining_log.get_suffix(lib_index))
    ids_path = fp_lib.id_files[lib_index]

    n_mols = read_np_data(mintd_path).shape[0]
    mintds = open_memmap(mintd_path, dtype=np.float32, mode='r', shape=(n_mols,))
    ids = open_memmap(ids_path, dtype=np.int64, mode='r', shape=(n_mols,))
    exclusions = open_memmap(excl_path, dtype=np.uint8, mode='r+', shape=(n_mols,))

    lib_array_indices = []
    selected_indices = []
    absolute_ids = []

    for start in range(0, n_mols, chunk_size):
        end = min(start + chunk_size, n_mols)
        chunk_mintds = np.array(mintds[start:end], dtype=np.float32, copy=True)
        chunk_bins = np.floor(chunk_mintds * 1000).astype(np.int64)
        chunk_excls = np.array(exclusions[start:end], dtype=np.uint8, copy=True)

        for i, (mintd_bin, excl) in enumerate(zip(chunk_bins, chunk_excls)):
            if not excl and mintd_bin <= bin_thresh:
                abs_index = start + i
                lib_array_indices.append((lib_index, abs_index))
                selected_indices.append(abs_index)
                absolute_ids.append(ids[abs_index])
                exclusions[abs_index] = 1  # set exclusion flag immediately

    exclusions.flush()  # commit new exclusions to disk

    if len(lib_array_indices) == 0:
        lib_array_indices = np.zeros((0, 2), dtype=np.int64)  # ← consistent shape
    else:
        lib_array_indices = np.asarray(lib_array_indices, dtype=np.int64)

    smi_list = fp_lib.load_smiles_indices(lib_index, selected_indices)
    return np.array(lib_array_indices, dtype=np.int64), smi_list, np.array(absolute_ids, dtype=np.int64)


@njit
def _get_todock_libarray_indices(lib_array_indices, mintds, exclusions, mintd_thresh, lib_index, n_todock):
    indices = np.zeros(len(mintds), dtype=np.int64)
    count = 0
    for i, mintd in enumerate(mintds):
        if mintd <= mintd_thresh:
            lib_array_indices[n_todock] = [lib_index, i]
            indices[count] = i
            count += 1
            n_todock += 1
            exclusions[i] = 1
    return n_todock, indices[:count]


def _add_exclusions_one_index(chaining_log, lib_index, array_indices, n):
    exclusions = np.zeros(n, dtype=np.uint8)
    exclusions[array_indices] = 1
    chaining_log.add_exclusions(exclusions, lib_index)

