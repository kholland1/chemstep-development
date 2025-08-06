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
    def __init__(self, fp_lib, chemstep_params, output_directory, n_proc, use_pickle=False, scores_fns=None,
                 ids_fns=None, verbose=False, skip_setup=False, write_df=False, df_name=None, scores_dir=None,
                 write_complete_info=False, complete_info_dir=None, docking_method="manual",
                 smi_id_prefix="CSLB", smi_id_offset=0, pickle_prefix=None, scheduler=None):
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
        self.ids_fns = ids_fns
        self.n_proc = n_proc
        self.verbose = verbose
        self.write_df = write_df
        self.write_complete_info = write_complete_info
        if self.write_complete_info:
            assert complete_info_dir is not None
        self.complete_info_dir = complete_info_dir
        if write_df:
            assert df_name is not None
            assert scores_dir is not None
        self.df_name = df_name
        self.scores_dir = scores_dir
        self.docking_method = docking_method
        self.smi_id_prefix = smi_id_prefix
        self.smi_id_offset = smi_id_offset
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
        self.used_beacons_fps = np.zeros((self.params.max_n_rounds * self.params.max_beacons,
                                          self.fp_lib.fp_length_bytes), dtype=np.uint8)
        self.used_beacons_count = 0

    def print_verbose(self, s):
        if self.verbose:
            print(s)

    def run_one_round(self, round_n, scores_dict):
        beacons = self.get_beacons(scores_dict, round_n)
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
            scores_dict = self.lookup_dock(lib_array_indices, smi_list, round_n)
        elif self.docking_method == "manual":
            self.write_smi_file(smi_list, lib_array_indices, round_n, absolute_ids)
            scores_dict = None
        else:
            raise ValueError(f"Docking method {self.docking_method} not yet implemented")

        if self.write_df:
            write_stats_df(self.scores_dir, self.chaining_log.log_folder, self.score_thresh, self.df_name)

        if self.use_pickle:
            with open(f'{self.pickle_prefix}_{round_n}.pickle', 'wb') as f:
                pickle.dump(self, f)
        return scores_dict

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
        if self.params.screen_novelty:
            self.screen_novelty()
        with open(self.params.seed_scores_file, 'rb') as f:
            scores_dict = pickle.load(f)
        if score_thresh is None:
            self.set_score_thresh(scores_dict)
        else:
            self.set_score_thresh(scores_dict)
            self.score_thresh = score_thresh
        for i in range(1, self.params.max_n_rounds+1):
            scores_dict = self.run_one_round(i, scores_dict)

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
                "account": "rrg-najmanov",
                "ntasks": "1",
                "mem": "4GB",
                "nodes": "1",
                "cpus-per-task": "1",
                "time": "12:00:00"
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
                for row in mintd_distrib:
                    f.write(f"{row[0]} {row[1]}\n")

        target_n = self.params.n_docked_per_round
        assert np.sum(mintd_distrib[:, 1]) > target_n

        count = 0
        mintd_thresh = None
        for row in mintd_distrib:
            count += row[1]
            if count >= target_n:
                mintd_thresh = row[0]
                break
        assert mintd_thresh is not None

        args = [(i, mintd_thresh, self.fp_lib, self.chaining_log) for i in range(self.fp_lib.n_files)]

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
        score_list = docker.get_score_list()
        self.print_verbose("SCORES_ROUND_{}: {}".format(round_n, score_list))
        self.print_verbose("SMILES_ROUND_{}: {}".format(round_n, smi_list))
        scores_dict = dict()
        full_ids, scores = [], []
        for (lib_index, arr_index), score in zip(lib_array_indices, score_list):
            full_id = self.fp_lib.get_full_index(lib_index, arr_index)
            if full_id in scores_dict:
                raise ValueError("This one already in: {} (from ({}, {}))".format(full_id, lib_index, arr_index))
            scores_dict[full_id] = score
            if score <= self.score_thresh:
                full_ids.append(full_id)
                scores.append(score)
        if self.write_complete_info:
            with open(f"{self.complete_info_dir}/hits_round_{round_n}.df", 'w') as f:
                f.write("full_id score\n")
                for full_id, score in zip(full_ids, scores):
                    f.write(f"{full_id} {score}\n")
        return scores_dict

    def set_score_thresh(self, seed_scores_dict):
        n = len(seed_scores_dict)
        scores = np.zeros(n)
        lib_arr_indices = np.zeros((n, 2), dtype=np.int64)
        for i, key in enumerate(seed_scores_dict):
            lib_arr_indices[i] = self.fp_lib.get_lib_array_indices(key)
            scores[i] = seed_scores_dict[key]
        self.score_thresh = np.quantile(scores, 10**(-1 * self.params.hit_pprop))

        lib_arr_dict = dict()
        for lib_i, arr_i in lib_arr_indices:
            if lib_i not in lib_arr_dict:
                lib_arr_dict[lib_i] = set()
            lib_arr_dict[lib_i].add(arr_i)

        for lib_i in lib_arr_dict:
            exclusions = np.zeros(self.fp_lib.lengths[lib_i], dtype=np.uint8)
            for arr_i in lib_arr_dict[lib_i]:
                exclusions[arr_i] = 1
            self.chaining_log.add_exclusions(exclusions, lib_i)

    def get_beacons(self, scores_dict, round_n):
        beacons_indices = []
        beacons_scores = []
        for index in scores_dict:
            if scores_dict[index] <= self.score_thresh:
                beacons_indices.append(index)
                beacons_scores.append(scores_dict[index])
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
        while len(kept_beacons) < self.params.max_beacons and np.sum(selected) < len(selected):
            max_index = np.argmax(distance_vector)
            selected[max_index] = 1
            kept_beacons.append(self.unused_beacons[max_index])
            distance_vector = np.minimum(distance_vector, 1 - get_tanimoto_max(np.array([all_fps[max_index]]), all_fps))
        self.unused_beacons = [x for i, x in enumerate(self.unused_beacons) if selected[i] == 0]
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


def _process_single_lib_chunked(lib_index, mintd_thresh, fp_lib, chaining_log, chunk_size=100_000):
    mintd_path = chaining_log.get_filename(chaining_log.mintd_prefix, chaining_log.get_suffix(lib_index))
    excl_path = chaining_log.get_filename(chaining_log.exclusion_prefix, chaining_log.get_suffix(lib_index))
    ids_path = fp_lib.id_files[lib_index]

    n_mols = read_np_data(mintd_path).shape[0]
    mintds = np.memmap(mintd_path, dtype=np.float32, mode='r', shape=(n_mols,))
    ids = np.memmap(ids_path, dtype=np.int64, mode='r', shape=(n_mols,))
    exclusions = np.memmap(excl_path, dtype=np.uint8, mode='r+', shape=(n_mols,))

    lib_array_indices = []
    selected_indices = []
    absolute_ids = []

    for start in range(0, n_mols, chunk_size):
        end = min(start + chunk_size, n_mols)
        chunk_mintds = mintds[start:end]
        chunk_excls = exclusions[start:end]

        for i, (mintd, excl) in enumerate(zip(chunk_mintds, chunk_excls)):
            if not excl and mintd <= mintd_thresh:
                abs_index = start + i
                lib_array_indices.append((lib_index, abs_index))
                selected_indices.append(abs_index)
                absolute_ids.append(ids[abs_index])
                exclusions[abs_index] = 1  # set exclusion flag immediately

    exclusions.flush()  # commit new exclusions to disk

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

