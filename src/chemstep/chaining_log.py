from chemstep.fp_library import FpLibrary
import numpy as np
from numba import njit
import os
from multiprocessing import Pool


class ChainingLog:
    """Bookkeeping for a ChemSTEP run across multiple rounds.

    Handles persistent storage of exclusion masks, minimum Tanimoto
    distance (minTD) arrays, and minTD distributions for each library
    chunk. Also manages job and log directories.

    Attributes:
        fp_library (FpLibrary): The fingerprint library being screened.
        log_folder (str): Path to the log directory.
        exclusion_prefix (str): Filename prefix for exclusion files.
        mintd_prefix (str): Filename prefix for minTD files.
        mintd_distrib_prefix (str): Filename prefix for minTD
            distribution files.
        jobs_folder (str): Path to the jobs directory.
    """

    def __init__(self, fp_library, log_folder, n_proc, exclusion_prefix="excl", mintd_prefix="mintds",
                 mintd_distrib_prefix="mintddistrib", write_empty_files=True,
                 jobs_folder=None):
        """Initialize a :class:`ChainingLog` instance.

        Args:
            fp_library (FpLibrary): Fingerprint library instance.
            log_folder (str): Path to the log directory.
            exclusion_prefix (str, optional): Prefix for exclusion files.
            mintd_prefix (str, optional): Prefix for minTD files.
            mintd_distrib_prefix (str, optional): Prefix for minTD
                distribution files.
            write_empty_files (bool, optional): If True, create empty
                log files on initialization.
            jobs_folder (str, optional): Path to the jobs folder. If
                None, defaults to ``<log_folder>/jobs``.
        """
        assert isinstance(fp_library, FpLibrary)
        self.fp_library = fp_library
        assert isinstance(log_folder, str)
        if log_folder.endswith('/'):
            log_folder = log_folder[:-1]
        self.log_folder = log_folder
        self.n_proc = n_proc
        self.exclusion_prefix = exclusion_prefix
        self.mintd_prefix = mintd_prefix
        self.mintd_distrib_prefix = mintd_distrib_prefix

        if jobs_folder is None:
            self.jobs_folder = "{}/jobs".format(self.log_folder)
        else:
            self.jobs_folder = jobs_folder
        if write_empty_files:
            self.write_empty_files()

    def write_empty_files(self):
        """Create empty exclusion, minTD, and minTD distribution files.

        Initializes all log files for each library chunk with the
        correct shape, dtype, and starting values:
        * Exclusions: zeros (uint8)
        * minTD: all set to 2.0 (float32)
        * minTD distributions: zeros (int64)
        """
        if not os.path.isdir(self.log_folder):
            os.mkdir(self.log_folder)
        if not os.path.isdir(self.jobs_folder):
            os.mkdir(self.jobs_folder)
        prefixes = [self.exclusion_prefix, self.mintd_prefix, self.mintd_distrib_prefix]
        shapes = [None, None, 1001]
        data_types = [np.uint8, np.float32, np.int64]
        init_vals = [0, 2, 0]
        args = []
        for i, suffix in enumerate(self.fp_library.suffixes):
            length = self.fp_library.lengths[i]
            for prefix, shape, data_type, init_val in zip(prefixes, shapes, data_types, init_vals):
                args.append((shape, length, init_val, data_type, self.get_filename(prefix, suffix)))
        with Pool(self.n_proc) as p:
            p.starmap(_write_one_empty_files_set, args)

    def load_exclusions(self, index):
        """Load the exclusion array for a library chunk.

        Args:
            index (int): Library chunk index.

        Returns:
            np.ndarray[uint8]: Boolean mask of excluded molecules.
        """
        self._check_index(index)
        filename = self.get_filename(self.exclusion_prefix, self.get_suffix(index))
        data = np.load(filename)
        return data

    def add_exclusions(self, exclusions, index):
        """Add exclusions for a library chunk.

        Args:
            exclusions (np.ndarray[bool]): Boolean array of new
                exclusions to add.
            index (int): Library chunk index.

        Raises:
            AssertionError: If the shape of ``exclusions`` does not
                match the stored exclusion array.
        """
        self._check_index(index)
        previous_exclusions = self.load_exclusions(index)
        assert previous_exclusions.shape == exclusions.shape
        exclusions = np.logical_or(previous_exclusions, exclusions)
        filename = self.get_filename(self.exclusion_prefix, self.get_suffix(index))
        np.save(filename, exclusions)

    def load_mintd_distrib(self, index):
        """Load the minTD distribution for a library chunk.

        Args:
            index (int): Library chunk index.

        Returns:
            np.ndarray[int64]: Histogram array of length 1001.
        """
        self._check_index(index)
        filename = self.get_filename(self.mintd_distrib_prefix, self.get_suffix(index))
        return np.load(filename)

    def load_global_mintd_distrib(self):
        """Load and sum minTD distributions across all library chunks.

        Returns:
            np.ndarray[int64]: Global histogram array of length 1001.
        """
        global_distrib = self.load_mintd_distrib(0)
        for index in range(1, self.fp_library.n_files):
            mintd_distrib = self.load_mintd_distrib(index)
            global_distrib += mintd_distrib
        return global_distrib

    def load_mintds(self, index):
        """Load the minTD array for a library chunk.

        Args:
            index (int): Library chunk index.

        Returns:
            np.ndarray[float32]: minTD values.
        """
        self._check_index(index)
        filename = self.get_filename(self.mintd_prefix, self.get_suffix(index))
        data = np.load(filename)
        return data

    def add_mintds(self, mintds, index, update_distrib=True):
        """Update the minTD array for a library chunk.

        Args:
            mintds (np.ndarray[float32]): New minTD values to merge.
            index (int): Library chunk index.
            update_distrib (bool, optional): If True, also recompute
                and save the minTD distribution.
        """
        self._check_index(index)
        filename = self.get_filename(self.mintd_prefix, self.get_suffix(index))
        data = np.load(filename)
        exclusions = self.load_exclusions(index)
        data = update_mintds(data, mintds, exclusions)
        np.save(filename, data)
        if update_distrib:
            filename = self.get_filename(self.mintd_distrib_prefix, self.get_suffix(index))
            distrib = get_mintd_distrib(data)
            np.save(filename, distrib)

    def _check_index(self, index):
        """Validate that a library chunk index is within bounds.

        Args:
            index (int): Library chunk index.

        Raises:
            AssertionError: If index is negative or >= number of files.
        """
        assert index >= 0
        assert index < self.fp_library.n_files

    def get_suffix(self, index):
        """Get the filename suffix for a library chunk.

        Args:
            index (int): Library chunk index.

        Returns:
            str: Filename suffix.
        """
        return self.fp_library.suffixes[index]

    def get_filename(self, prefix, suffix, ext=".npy"):
        """Build a full path for a log file.

        Args:
            prefix (str): Filename prefix.
            suffix (str): Filename suffix.
            ext (str, optional): File extension. Defaults to ``.npy``.

        Returns:
            str: Full file path.
        """
        return "{}/{}{}{}".format(self.log_folder, prefix, suffix, ext)


def save_flush_np(data, filename):
    """Save a NumPy array to disk and flush buffers.

    Args:
        data (np.ndarray): Array to save.
        filename (str): Destination filename (``.npy`` appended if
            missing).
    """
    if not filename.endswith('.npy'):
        filename += ".npy"
    with open(filename, 'wb') as f:
        np.save(f, data)
        f.flush()  # flushes the internal Python buffer to the OS
        os.fsync(f.fileno())  # asks the OS to flush its buffers to disk


@njit
def update_mintds(data, mintds, exclusions):
    """Merge new minTD values into an existing minTD array.

    For each element:
        * If excluded, set to 2.0.
        * Else, set to the minimum of existing and new minTD.

    Args:
        data (np.ndarray[float32]): Existing minTD values.
        mintds (np.ndarray[float32]): New minTD values.
        exclusions (np.ndarray[bool]): Exclusion mask.

    Returns:
        np.ndarray[float32]: Updated minTD values.
    """
    assert data.shape[0] == mintds.shape[0] == exclusions.shape[0]
    for i in range(data.shape[0]):
        if exclusions[i]:
            data[i] = 2
            continue
        mintd = mintds[i]
        if mintd < data[i]:
            data[i] = mintd
    return data


@njit
def get_mintd_distrib(mintds):
    """Compute a histogram of minTD values.

    Values > 1 are ignored because they mean the molecule has been docked before.
    Remaining values are binned into 1001
    bins from 0.000 to 1.000 (inclusive).

    Args:
        mintds (np.ndarray[float32]): minTD values.

    Returns:
        np.ndarray[int64]: Histogram array of length 1001.
    """
    distrib = np.zeros(1001, dtype=np.int64)
    for mintd in mintds:
        if mintd > 1:
            continue
        index = int(np.floor(mintd*1000))
        distrib[index] += 1
    return distrib


def _write_one_empty_files_set(shape, length, init_val, data_type, filename):
    if shape is None:
        shape = length
    if init_val != 0:
        data = np.full(shape, init_val, dtype=data_type)
    else:
        data = np.zeros(shape, dtype=data_type)
    np.save(filename, data)

