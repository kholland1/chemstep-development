import numpy as np
from chemstep.fp_library import FpLibrary
from chemstep.parameters import CSParams, read_param_file
from chemstep.chaining_log import ChainingLog
from chemstep.search_job import SearchJob
from chemstep.lookup_docking import LookupDocking
from chemstep.stats import write_stats_df
import pickle
from multiprocessing import Pool
from numba import njit
from chemstep.fingerprints import get_tanimoto_max
import os
from chemstep.job_array import SlurmJobArray, SGEJobArray
from chemstep.utils import read_np_data
from chemstep.id_helper import int64_to_char, char_to_int64
from numpy.lib.format import open_memmap
from chemstep.bookkeeper import Bookkeeper


def load_from_pickle(fn):
    with open(fn, 'rb') as f:
        obj = pickle.load(f)
        assert isinstance(obj, CSAlgo)
        return obj


class CSAlgo:
    """ Main class for the ChemSTEP algorithm. Parameters are set, jobs are created, results are pooled,
        basically everything is handled by a single CSAlgo object. The object can be built from a previously pickled
        representation, which makes restarting/continuing easier.

        Attributes:
            fp_lib (FpLibrary): The library of considered molecules (precomputed fingerprints as NumPy arrays)
            params (CSParams): The parameters for this run of ChemSTEP
            chaining_log (ChainingLog): The object doing the bookkeeping as the rounds progress
    """
    def __init__(self, fp_lib, chemstep_params, output_directory, n_proc, use_pickle=True,
                 pickle_prefix="chemstep_algo", verbose=False, skip_setup=False, write_complete_info=True,
                 complete_info_dir=None, docking_method="manual", smi_id_prefix="CSLB", scheduler=None,
                 scores_fns=None):
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
        self.write_complete_info = write_complete_info
        if self.write_complete_info:
            if complete_info_dir is None:
                complete_info_dir = os.path.join(output_directory, "complete_info")
            os.makedirs(complete_info_dir, exist_ok=True)
        self.complete_info_dir = complete_info_dir
        self.docking_method = docking_method
        self.smi_id_prefix = smi_id_prefix
        self.book = Bookkeeper(self.complete_info_dir, self.smi_id_prefix)
        if scheduler is not None:
            if scheduler not in ["slurm", "sge"]:
                raise ValueError("Scheduler must be either 'slurm' or 'sge'")
        self.scheduler = scheduler
        self.print_verbose("about to setup ChainingLog")
        if skip_setup:
            self.chaining_log = ChainingLog(self.fp_lib, output_directory, write_empty_files=False)
        else:
            self.chaining_log = ChainingLog(self.fp_lib, output_directory)
        self.print_verbose("ChainingLog set")
        self.score_thresh = None
        self.unused_beacons = []
        self.current_beacons = []
        self.current_beacons_dists = []
        self.current_mintd_thresh = None
        self.used_beacons_fps = np.zeros((self.params.max_n_rounds * self.params.max_beacons,
                                          self.fp_lib.fp_length_bytes), dtype=np.uint8)
        self.used_beacons_count = 0

    def print_verbose(self, s):
        if self.verbose:
            print(s)

    def run_one_round(self, round_n, new_indices, new_scores):
        if round_n > 1:
            n_hits = int(np.sum(new_scores <= self.score_thresh))
            beacon_ids = [b[1] for b in self.current_beacons]
            beacon_scores = [b[0] for b in self.current_beacons]
            self.book.log_round(round_n-1, len(new_indices), n_hits, self.current_mintd_thresh,
                                beacon_ids, beacon_scores, self.current_beacons_dists)
        beacons = self.get_beacons(new_indices, new_scores, round_n)
        self.print_verbose(f"Starting round {round_n} with {len(beacons)} beacons")
        jobs = []
        for j in range(self.fp_lib.n_files):
            unique_id = "{}_{}".format(round_n, j)
            job = SearchJob(unique_id, beacons, round_n, self.fp_lib, self.chaining_log, j, scheduler=self.scheduler)
            jobs.append(job)
        if self.scheduler == "slurm":
            self.run_slurm_array(jobs, round_n)
        elif self.scheduler == "sge":
            self.run_sge_array(jobs, round_n)
        else:
            self.run_local(jobs)
        self.print_verbose("Starting docking for round {}".format(round_n))
        lib_array_indices, smi_list, absolute_ids = self.get_todock_list(round_n)
        self.used_beacons_fps[self.used_beacons_count:self.used_beacons_count+len(beacons)] = beacons
        self.used_beacons_count += len(beacons)

        if self.docking_method == "lookup":
            new_indices, new_scores = self.lookup_dock(lib_array_indices, smi_list, round_n)
        elif self.docking_method == "manual":
            self.write_smi_file(smi_list, lib_array_indices, round_n, absolute_ids)
            new_indices, new_scores = None, None
        else:
            raise ValueError(f"Docking method {self.docking_method} not yet implemented")

        if self.use_pickle:
            with open(f'{self.pickle_prefix}_{round_n}.pickle', 'wb') as f:
                pickle.dump(self, f)
        return new_indices, new_scores

    def write_smi_file(self, smi_list, lib_array_indices, round_n, absolute_ids):
        abs_out = open(f'{self.complete_info_dir}/absolute_ids_round_{round_n}.txt', 'w')
        with open("{}/smi_round_{}.smi".format(self.complete_info_dir, round_n), 'w') as f:
            for smi, lib_arr, abs_id in zip(smi_list, lib_array_indices, absolute_ids):
                full_index = self.fp_lib.get_full_index(lib_arr[0], lib_arr[1])
                char_name = int64_to_char(full_index, prefix=self.smi_id_prefix)
                f.write(f'{smi} {char_name}\n')
                abs_out.write(f'{abs_id}\n')
        abs_out.close()

    def linking_loop(self, score_thresh=None):
        seed_indices = np.load(self.params.seed_indices_file)
        seed_scores = np.load(self.params.seed_scores_file)
        if score_thresh is None:
            self.set_score_thresh(seed_indices, seed_scores)
            self.print_verbose(f"Automatically set score threshold to {self.score_thresh:.2f} " +
                               f"(pProp of {self.params.hit_pprop})")
        else:
            self.set_score_thresh(seed_indices, seed_scores)  # important to remove already docked compounds
            self.print_verbose(f"Automatically set score threshold to {self.score_thresh:.2f} " +
                               f"(pProp of {self.params.hit_pprop})")
            self.score_thresh = score_thresh
            self.print_verbose(f"Overrode score threshold to {self.score_thresh:.2f}")
        new_indices = seed_indices
        new_scores = seed_scores
        for i in range(1, self.params.max_n_rounds+1):
            new_indices, new_scores = self.run_one_round(i, new_indices, new_scores)

    def run_local(self, jobs):
        p = Pool(self.n_proc)
        p.map(_run_one_job_local, jobs)

    def run_slurm_array(self, jobs, round_n):
        for j in range(self.fp_lib.n_files):
            pickle_path = os.path.join(self.chaining_log.jobs_folder, f"{round_n}_{j}.pickle")
            with open(pickle_path, 'wb') as f:
                pickle.dump(jobs[j], f)

        job_array = SlurmJobArray(
            round_n=round_n,
            n_jobs=len(jobs),
            job_folder=self.chaining_log.jobs_folder,
            python_exec="python",
            slurm_options={
                "account": "rrg-mailhoto",
                "ntasks": "1",
                "mem": "4GB",
                "nodes": "1",
                "cpus-per-task": "1",
                "time": "1:00:00"
            }
        )
        job_id = job_array.submit()
        job_array.wait(job_id)

    def run_sge_array(self, jobs, round_n):
        job_array = SGEJobArray(
            round_n=round_n,
            n_jobs=len(jobs),
            job_folder=self.chaining_log.jobs_folder,
            python_exec="python",
            sge_options={
                "l": "h_rt=12:00:00",
                "pe": "smp 1",
                "cwd": None
            }
        )
        job_id = job_array.submit()
        job_array.wait(job_id)

    def get_todock_list(self, round_n):
        mintd_distrib = self.chaining_log.load_global_mintd_distrib()
        if self.write_complete_info:
            with open(f"{self.complete_info_dir}/mintd_distrib_{round_n}.df", 'w') as f:
                f.write("mintd count\n")
                for mintd_bin, n in enumerate(mintd_distrib):
                    f.write(f"{(mintd_bin + 0.5) / 1000} {n}\n")

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

        args = [(i, mintd_bin_thresh, self.fp_lib, self.chaining_log) for i in range(self.fp_lib.n_files)]

        from multiprocessing import Pool
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

    def lookup_dock(self, lib_array_indices, smi_list, round_n):
        self.print_verbose("About to start 'docking'")
        docker = LookupDocking(lib_array_indices, smi_list, self.scores_fns, self.fp_lib, verbose=self.verbose)
        docker.dock_all()
        self.print_verbose("'Docking' done")
        new_scores = docker.get_scores_list()
        new_indices = np.zeros(len(new_scores), dtype=np.int64)
        self.print_verbose("SCORES_ROUND_{}: {}".format(round_n, new_scores))
        self.print_verbose("SMILES_ROUND_{}: {}".format(round_n, smi_list))

        for i, ((lib_index, arr_index), score) in enumerate(zip(lib_array_indices, new_scores)):
            full_id = self.fp_lib.get_full_index(lib_index, arr_index)
            new_indices[i] = full_id
        if self.write_complete_info:
            with open(f"{self.complete_info_dir}/hits_round_{round_n}.df", 'w') as f:
                f.write("full_id score\n")
                for full_id, score in zip(new_indices, new_scores):
                    f.write(f"{full_id} {score}\n")
        return new_indices, new_scores

    def set_score_thresh(self, seed_indices, seed_scores):
        lib_arr_indices = np.zeros((len(seed_scores), 2), dtype=np.int64)
        for i, seed_index in enumerate(seed_indices):
            lib_arr_indices[i] = self.fp_lib.get_lib_array_indices(seed_index)
        self.score_thresh = np.quantile(seed_scores, 10**(-1 * self.params.hit_pprop))

        lib_arr_dict = dict()
        for lib_i, arr_i in lib_arr_indices:
            if lib_i not in lib_arr_dict:
                lib_arr_dict[lib_i] = set()
            lib_arr_dict[lib_i].add(arr_i)

        for lib_i in lib_arr_dict:
            exclusions = np.zeros(self.fp_lib.lengths[lib_i], dtype=np.uint8)
            # TODO: make chunked
            for arr_i in lib_arr_dict[lib_i]:
                exclusions[arr_i] = 1
            self.chaining_log.add_exclusions(exclusions, lib_i)

    def get_beacons(self, new_indices, new_scores, round_n):
        """
        Select a diversity-pruned set of beacons from the latest docking results.

        Parameters
        ----------
        new_indices : np.ndarray[int64]
            Full indices (FpLibrary format) for the newly docked molecules.
        new_scores : np.ndarray[float32]
            Corresponding docking scores.
        round_n : int
            Current chaining round number (0-based).

        Returns
        -------
        np.ndarray[uint8]
            Fingerprint array (≤ max_beacons rows) for the beacons to use
            in the next search round.
        """
        # 1. Add newly qualified hits to the pool of candidate beacons
        beacons_candidates = [
            (score, idx)
            for idx, score in zip(new_indices, new_scores)
            if score <= self.score_thresh
        ]
        self.unused_beacons.extend(beacons_candidates)
        self.unused_beacons.sort()  # ascending score

        selected_fps = self.apply_beacons_diversity()

        if self.write_complete_info:
            outfile = f"{self.complete_info_dir}/beacons_round_{round_n}.df"
            with open(outfile, "w") as f:
                f.write("index score\n")
                for score, full_id in self.current_beacons:  # len(kept_pairs) == len(selected_fps)
                    f.write(f"{full_id} {score}\n")

        return selected_fps

    def bak_get_beacons(self, new_indices, new_scores, round_n):
        beacons_indices = []
        beacons_scores = []
        for index, score in zip(new_indices, new_scores):
            if score <= self.score_thresh:
                beacons_indices.append(index)
                beacons_scores.append(score)
        self.unused_beacons += [x for x in zip(beacons_scores, beacons_indices)]
        self.unused_beacons = sorted(self.unused_beacons)
        filtered_fps = self.apply_beacons_diversity()
        if len(self.unused_beacons) >= self.params.max_beacons:
            n_beacons = self.params.max_beacons
        else:
            n_beacons = len(self.unused_beacons)

        beacons = filtered_fps[:n_beacons]
        if self.write_complete_info:
            with open(f"{self.complete_info_dir}/beacons_round_{round_n}.df", 'w') as f:
                f.write("index score\n")
                for score, index in self.unused_beacons[:n_beacons]:
                    f.write(f"{index} {score}\n")
        self.unused_beacons = self.unused_beacons[n_beacons:]
        return beacons

    def screen_novelty(self):
        raise ValueError("screen_novelty() not yet implemented")

    def apply_beacons_diversity(self):
        return self.apply_beacons_diversity_maxdiv()

    def load_all_fps_unused_beacons(self):
        # fingerprints for all unused beacons
        all_fps = np.zeros((len(self.unused_beacons), self.fp_lib.fp_length_bytes), dtype=np.uint8)

        # dict of dicts to untangle library indices
        lib_arr_dict = dict()

        for i, (_, full_index) in enumerate(self.unused_beacons):
            lib_index, arr_index = self.fp_lib.get_lib_array_indices(full_index)
            if lib_index not in lib_arr_dict:
                lib_arr_dict[lib_index] = dict()
            lib_arr_dict[lib_index][arr_index] = i

        for lib_index in lib_arr_dict:
            fps = self.fp_lib.load_fps(lib_index)
            for arr_index in lib_arr_dict[lib_index]:
                all_fps[lib_arr_dict[lib_index][arr_index]] = fps[arr_index]
        return all_fps

    def apply_beacons_diversity_maxdiv(self):
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
        self.unused_beacons = [x for i, x in enumerate(self.unused_beacons) if selected[i] == 0]
        self.current_beacons = kept_beacons
        self.current_beacons_dists = distances
        return all_fps[selected == 1]

    def all_jobs_completed(self, round_n):
        for j in range(self.fp_lib.n_files):
            pickle_path = os.path.join(self.chaining_log.jobs_folder, f"{round_n}_{j}.pickle")
            if not os.path.exists(pickle_path):
                return False
            with open(pickle_path, 'rb') as f:
                job = pickle.load(f)
            if not getattr(job, 'completed', False):
                return False
        return True


def _run_one_job_local(job):
    job.run_local()


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
        chunk_mintds = mintds[start:end]
        chunk_bins = np.floor(chunk_mintds * 1000).astype(np.int64)
        chunk_excls = exclusions[start:end]

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

