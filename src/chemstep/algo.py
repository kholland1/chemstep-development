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
from bksltk.fingerprints import get_tc, get_tanimoto_max


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
            params (CSParams): The parameters for this run of CSC
            chaining_log (ChainingLog): The object doing the bookkeeping as the rounds progress
    """
    def __init__(self, fp_lib, chemstep_params, output_directory, n_proc, use_pickle=False, scores_fns=None,
                 ids_fns=None, verbose=False, skip_setup=False, write_df=False, df_name=None, scores_dir=None,
                 hit_score_thresh=None, write_complete_info=False, complete_info_dir=None, docking_method="manual",
                 use_maxdiv=False, smi_id_prefix="ZWIT", smi_id_offset=0, smi_id_nzeros=9, pickle_prefix=None):
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
            assert hit_score_thresh is not None
        self.df_name = df_name
        self.scores_dir = scores_dir
        self.hit_score_thresh = hit_score_thresh
        self.docking_method = docking_method
        self.use_maxdiv = use_maxdiv
        self.smi_id_prefix = smi_id_prefix
        self.smi_id_offset = smi_id_offset
        self.smi_id_nzeros = smi_id_nzeros
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
        self.print_verbose("Starting round {}".format(round_n))
        jobs = []
        for j in range(self.fp_lib.n_files):
            unique_id = "{}_{}".format(round_n, j)
            job = SearchJob(unique_id, beacons, round_n, self.fp_lib, self.chaining_log, j)
            jobs.append(job)
        self.run_local(jobs)
        self.print_verbose("Starting docking for round {}".format(round_n))
        lib_array_indices, smi_list = self.get_todock_list(round_n)
        self.used_beacons_fps[self.used_beacons_count:self.used_beacons_count+len(beacons)] = beacons
        self.used_beacons_count += len(beacons)

        if self.docking_method == "lookup":
            scores_dict = self.lookup_dock(lib_array_indices, smi_list, round_n)
        elif self.docking_method == "manual":
            self.write_smi_file(smi_list, lib_array_indices, round_n)
            scores_dict = None
        else:  # TODO: allow other docking methods
            raise ValueError("Docking method {} not yet implemented".format(self.docking_method))

        if self.write_df:
            write_stats_df(self.scores_dir, self.chaining_log.log_folder, self.hit_score_thresh, self.df_name)

        if self.use_pickle:
            with open(f'{self.pickle_prefix}_{round_n}.pickle', 'wb') as f:
                pickle.dump(self, f)
        return scores_dict

    def write_smi_file(self, smi_list, lib_array_indices, round_n):
        with open("{}/smi_round_{}.smi".format(self.complete_info_dir, round_n), 'w') as f:
            for smi, lib_arr in zip(smi_list, lib_array_indices):
                full_index = self.fp_lib.get_full_index(lib_arr[0], lib_arr[1])
                f.write(f'{smi} {self.smi_id_prefix}{full_index+self.smi_id_offset:0{self.smi_id_nzeros}d}\n')

    def linking_loop(self, score_thresh=None):
        if self.params.screen_novelty:
            self.screen_novelty()
        with open(self.params.seed_scores_file, 'rb') as f:
            scores_dict = pickle.load(f)
        if score_thresh is None:
            self.set_score_thresh(scores_dict)
        else:
            self.hit_score_thresh = score_thresh
        for i in range(1, self.params.max_n_rounds+1):
            scores_dict = self.run_one_round(i, scores_dict)

    def run_local(self, jobs):
        p = Pool(self.n_proc)
        p.map(_run_one_job_local, jobs)

    def run_slurm_array(self, jobs):
        # TODO: implement
        pass

    def run_sge_array(self, jobs):
        # TODO: implement
        pass

    def get_todock_list(self, round_n):
        mintd_distrib = self.chaining_log.load_global_mintd_distrib(round_n)
        if self.write_complete_info:
            with open("{}/mintd_distrib_{}.df".format(self.complete_info_dir, round_n), 'w') as f:
                f.write("mintd count\n")
                for row in mintd_distrib:
                    f.write("{} {}\n".format(row[0], row[1]))
        target_n = self.params.n_docked_per_round
        count = 0
        assert np.sum(mintd_distrib[:, 1]) > target_n
        mintd_thresh = None
        for row in mintd_distrib:
            count += row[1]
            if count > target_n:
                mintd_thresh = row[0]
                break
        assert mintd_thresh is not None
        count = int(round(count))
        lib_array_indices = np.zeros((count, 2), dtype=np.int64)  # each row: [lib_index, array_index]
        n_todock = 0
        smi_list = []
        for lib_index in range(self.fp_lib.n_files):  # TODO: make parallel ( if necessary)
            mintds = self.chaining_log.load_mintds(lib_index, round_n)
            exclusions = self.chaining_log.load_exclusions(lib_index, round_n - 1)
            n_todock, indices = _get_todock_libarray_indices(lib_array_indices, mintds, exclusions, mintd_thresh,
                                                             lib_index, n_todock)
            smiles = self.fp_lib.load_smiles_indices(lib_index, indices)
            smi_list += [x for x in smiles]
            self.chaining_log.add_exclusions(exclusions, lib_index, round_n)
            self.print_verbose("Getting to_dock list, lib_index {}".format(lib_index))
        lib_array_indices = lib_array_indices[:n_todock]
        return lib_array_indices, smi_list

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
            if score <= self.hit_score_thresh:
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
            self.chaining_log.add_exclusions(exclusions, lib_i, 0)

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
        if self.use_maxdiv:
            return self.apply_beacons_diversity_maxdiv()
        else:
            return self.apply_beacons_diversity_distthresh()

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
        # TODO: test this
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

    def apply_beacons_diversity_distthresh(self, use_previous_beacons=True):
        dist_thresh = self.params.diversity_dist_thresh

        all_fps = self.load_all_fps_unused_beacons()

        if dist_thresh == 0:
            return all_fps
        else:
            kept_fps = np.zeros((len(all_fps), self.fp_lib.fp_length_bytes), dtype=np.uint8)
            count = 1
            kept_fps[0] = all_fps[0]
            kept_beacons = [self.unused_beacons[0]]
            for i in range(1, len(all_fps)):
                curr_fp = all_fps[i]
                mindist = 2
                for j in range(count):
                    dist = 1 - get_tc(curr_fp, kept_fps[j])
                    if dist < mindist:
                        mindist = dist
                for j in range(self.used_beacons_count):
                    dist = 1 - get_tc(curr_fp, self.used_beacons_fps[j])
                    if dist < mindist:
                        mindist = dist
                if mindist >= dist_thresh:
                    kept_fps[count] = curr_fp
                    kept_beacons.append(self.unused_beacons[i])
                    count += 1
            self.unused_beacons = kept_beacons  # TODO: this seems wrong? Mostly using maxdiv for now so doesn't matter
            return kept_fps


def _run_one_job_local(job):
    job.run_local()


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

