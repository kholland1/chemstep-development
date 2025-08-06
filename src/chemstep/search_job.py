from chemstep.fp_library import FpLibrary
from chemstep.chaining_log import ChainingLog
from chemstep.fingerprints import get_tanimoto_max_excl
from chemstep.utils import read_np_data
import numpy as np
import pickle
import os


def run_from_pickle(pickle_path):
    """
    Load a SearchJob from disk, run it, mark it completed, and overwrite the pickle.
    """
    with open(pickle_path, 'rb') as f:
        job = pickle.load(f)
    assert isinstance(job, SearchJob)

    job.run_job()

    with open(pickle_path, 'wb') as f:
        pickle.dump(job, f)


class SearchJob:
    """ Class representing a search job across a subset of the library, from a list of beacons. Can either run locally
        or generate a Slurm/other job that gets submitted to a scheduler.
    """

    def __init__(self, unique_id, beacons_array, round_n, fp_library, chaining_log, library_index, scheduler=None,
                 chunk_size=50000):
        assert isinstance(fp_library, FpLibrary)
        assert isinstance(chaining_log, ChainingLog)
        self.unique_id = unique_id
        self.beacons = beacons_array
        self.round_n = round_n
        self.lib = fp_library
        self.lib_index = library_index
        self.chaining_log = chaining_log
        self.scheduler = scheduler
        self.chunk_size = chunk_size
        self.completed = False

    def run_local(self):
        self.run_job()

    def run_job(self):
        """
        Run the job over a single fingerprint file in chunks, updating mintd values.

        This function performs memory-efficient read/write using np.memmap.
        It can be called from run_local() or Slurm/SGE job wrappers.

        """
        fp_path = self.lib.fp_files[self.lib_index]
        excl_path = self.chaining_log.get_filename(self.chaining_log.exclusion_prefix,
                                                   self.chaining_log.get_suffix(self.lib_index))
        mintd_path = self.chaining_log.get_filename(self.chaining_log.mintd_prefix,
                                                    self.chaining_log.get_suffix(self.lib_index))

        n_mols, fp_len = read_np_data(fp_path).shape

        fps = np.memmap(fp_path, dtype=np.uint8, mode='r', shape=(n_mols, fp_len))
        exclusions = np.memmap(excl_path, dtype=np.uint8, mode='r', shape=(n_mols,))
        mintds = np.memmap(mintd_path, dtype=np.float32, mode='r+', shape=(n_mols,))

        for start in range(0, n_mols, self.chunk_size):
            end = min(start + self.chunk_size, n_mols)
            fp_chunk = fps[start:end]
            excl_chunk = exclusions[start:end]
            mintd_chunk = 1 - get_tanimoto_max_excl(self.beacons, fp_chunk, excl_chunk)
            mintds[start:end] = np.minimum(mintds[start:end], mintd_chunk)

        mintds.flush()
        self.completed = True
