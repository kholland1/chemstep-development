from chemstep.fp_library import FpLibrary
import numpy as np
from numba import njit
import os


class ChainingLog:
    """ Class that does all the necessary bookkeeping for an instance of ChemSTEP as it runs through several rounds.
    """

    def __init__(self, fp_library, log_folder, max_n_rounds, exclusion_prefix="excl", mintd_prefix="mintds",
                 mintd_distrib_prefix="mintddistrib", best_beacon_prefix="bestbeacon", write_empty_files=True,
                 jobs_folder=None):
        assert isinstance(fp_library, FpLibrary)
        self.fp_library = fp_library
        assert isinstance(log_folder, str)
        if log_folder.endswith('/'):
            log_folder = log_folder[:-1]
        self.log_folder = log_folder
        self.max_n_rounds = max_n_rounds
        self.exclusion_prefix = exclusion_prefix
        self.mintd_prefix = mintd_prefix
        self.mintd_distrib_prefix = mintd_distrib_prefix
        self.best_beacon_prefix = best_beacon_prefix
        if jobs_folder is None:
            self.jobs_folder = "{}/jobs".format(self.log_folder)
        else:
            self.jobs_folder = jobs_folder
        if write_empty_files:
            self.write_empty_files()

    def write_empty_files(self):
        if not os.path.isdir(self.log_folder):
            os.mkdir(self.log_folder)
        if not os.path.isdir(self.jobs_folder):
            os.mkdir(self.jobs_folder)
        prefixes = [self.exclusion_prefix, self.mintd_prefix, self.mintd_distrib_prefix, self.best_beacon_prefix]
        shapes = [[None, self.max_n_rounds+1], [None, self.max_n_rounds+1], [1001, self.max_n_rounds+1],
                  [None, self.max_n_rounds+1]]
        data_types = [np.uint8, np.float32, np.int32, np.int64]
        init_vals = [0, 2, 0, -1]
        for i, suffix in enumerate(self.fp_library.suffixes):
            length = self.fp_library.lengths[i]
            for prefix, shape, data_type, init_val in zip(prefixes, shapes, data_types, init_vals):
                for j, elem in enumerate(shape):
                    if elem is None:
                        shape[j] = length
                if init_val != 0:
                    data = np.full(shape, init_val, dtype=data_type)
                else:
                    data = np.zeros(shape, dtype=data_type)
                np.save(self.get_filename(prefix, suffix), data)

    def load_exclusions(self, index, round_n):
        self._check_params(index, round_n)
        filename = self.get_filename(self.exclusion_prefix, self.get_suffix(index))
        data = np.load(filename)
        return data[:, round_n]

    def add_exclusions(self, exclusions, index, round_n):
        self._check_params(index, round_n)
        filename = self.get_filename(self.exclusion_prefix, self.get_suffix(index))
        data = np.load(filename)
        assert np.sum(data[:, round_n]) == 0
        data[:, round_n] = exclusions
        np.save(filename, data)

    def load_mintd_distrib(self, index, round_n):
        self._check_params(index, round_n)
        filename = self.get_filename(self.mintd_distrib_prefix, self.get_suffix(index))
        data = np.load(filename)
        td_values = np.arange(0, stop=1.001, step=0.001)
        distrib = np.zeros((1001, 2))
        distrib[:, 0] = td_values
        distrib[:, 1] = data[:, round_n]
        return distrib

    def load_global_mintd_distrib(self, round_n):
        global_distrib = self.load_mintd_distrib(0, round_n)
        for index in range(1, self.fp_library.n_files):
            mintd_distrib = self.load_mintd_distrib(index, round_n)
            global_distrib[:, 1] += mintd_distrib[:, 1]
        return global_distrib

    def load_mintds(self, index, round_n):
        self._check_params(index, round_n)
        filename = self.get_filename(self.mintd_prefix, self.get_suffix(index))
        data = np.load(filename)
        return data[:, round_n]

    def add_mintds(self, mintds, index, round_n, update_distrib=True):
        self._check_params(index, round_n)
        filename = self.get_filename(self.mintd_prefix, self.get_suffix(index))
        data = np.load(filename)
        data = update_mintds(data, mintds, round_n)
        np.save(filename, data)
        if update_distrib:
            filename = self.get_filename(self.mintd_distrib_prefix, self.get_suffix(index))
            mintd_distribs = np.load(filename)
            distrib = get_mintd_distrib(data[:, round_n])
            mintd_distribs[:, round_n] = distrib
            np.save(filename, mintd_distribs)
            
    def load_best_beacons(self):
        pass
        # TODO: complete this

    def add_best_beacons(self):
        pass
        # TODO: complete this

    def _check_params(self, index, round_n):
        assert index >= 0
        assert round_n >= 0
        assert round_n <= self.max_n_rounds
        assert index < self.fp_library.n_files

    def get_suffix(self, index):
        return self.fp_library.suffixes[index]

    def get_filename(self, prefix, suffix, ext=".npy"):
        return "{}/{}{}{}".format(self.log_folder, prefix, suffix, ext)


def save_flush_np(data, filename):
    if not filename.endswith('.npy'):
        filename += ".npy"
    with open(filename, 'wb') as f:
        np.save(f, data)
        f.flush()  # flushes the internal Python buffer to the OS
        os.fsync(f.fileno())  # asks the OS to flush its buffers to disk


@njit
def update_mintds(data, mintds, round_n):
    assert data.shape[0] == mintds.shape[0]
    for i in range(data.shape[0]):
        mintd = mintds[i]
        if mintd > 1:  # this means exclusion in previous rounds
            continue
        last_mintd = data[i, round_n - 1]
        if mintd < last_mintd:
            data[i, round_n] = mintd
        else:
            data[i, round_n] = last_mintd
    return data


@njit
def get_mintd_distrib(mintds):
    distrib = np.zeros(1001, dtype=np.int32)
    for mintd in mintds:
        if mintd > 1:
            continue
        index = int(np.round(mintd*1000))
        distrib[index] += 1
    return distrib

