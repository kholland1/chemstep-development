import numpy as np
from chemstep.utils import read_np_data
from numba import njit
import pickle


def load_library_from_pickle(fn):
    """Load a previously pickled :class:`FpLibrary` object from disk.

    This is the recommended way to restore a saved fingerprint
    library, for example when resuming a screening run.

    Args:
        fn (str): Path to the pickle file containing the saved
            :class:`FpLibrary` instance.

    Returns:
        FpLibrary: The unpickled fingerprint library object.

    Raises:
        AssertionError: If the unpickled object is not an instance of
            :class:`FpLibrary`.

    Notes:
        The pickle stores only file paths to the underlying library
        data. For maximal interoperability, ensure that you either load it from
        the same environment where it was created, or that the paths to the
        files were absolute.
    """
    with open(fn, 'rb') as f:
        obj = pickle.load(f)
        assert isinstance(obj, FpLibrary)
        return obj


class FpLibrary:
    """A library of molecules represented as precomputed fingerprints.

    Provides methods for validating the library structure and loading
    fingerprints, IDs, and SMILES strings.

    Attributes:
        n_files (int): Number of fingerprint/ID/SMILES file triplets.
        fp_files (list[str]): Paths to fingerprint (.npy) files.
        id_files (list[str]): Paths to ID (.npy) files (dtype=int64).
        smi_files (list[str]): Paths to SMILES (.smi or similar) files.
        fp_prefix (str): Filename prefix for fingerprint files.
        id_prefix (str): Filename prefix for ID files.
        smi_prefix (str): Filename prefix for SMILES files.
        suffixes (list[str]): Filename suffixes (excluding extension)
            for each library chunk.
        extensions (list[str]): File extensions for each library chunk.
        lengths (np.ndarray[int64]): Number of molecules per file.
        n_mols (int): Total number of molecules in the library.
        fp_length_bytes (int): Number of bytes per fingerprint.
    """

    def __init__(self, fingerprint_files, id_files, smi_files, outname, fp_prefix="fps", id_prefix="zids",
                 smi_prefix="smi"):
        """Initialize and validate an :class:`FpLibrary` instance.

        Args:
            fingerprint_files (list[str]): Paths to fingerprint files.
            id_files (list[str]): Paths to ID files.
            smi_files (list[str]): Paths to SMILES files.
            outname (str): Output pickle filename (``.pickle`` appended
                if missing).
            fp_prefix (str, optional): Prefix for fingerprint files.
            id_prefix (str, optional): Prefix for ID files.
            smi_prefix (str, optional): Prefix for SMILES files.

        Raises:
            AssertionError: If file list lengths mismatch or
                validation fails.
            ValueError: If naming conventions or file contents are
                inconsistent.
        """
        assert len(fingerprint_files) == len(id_files)
        self.n_files = len(fingerprint_files)
        self.fp_files = fingerprint_files
        self.id_files = id_files
        self.smi_files = smi_files
        self.fp_prefix = fp_prefix
        self.id_prefix = id_prefix
        self.smi_prefix = smi_prefix
        self.suffixes = []
        self.extensions = []
        self.lengths = []
        self.n_mols = None
        self.fp_length_bytes = None
        self._validate()
        if not outname.endswith('.pickle'):
            outname += '.pickle'
        with open(outname, 'wb') as f:
            pickle.dump(self, f)

    def _validate(self):
        """Validate the library’s file naming and contents.

        Checks:
            * Filenames match expected prefixes and suffixes.
            * Corresponding fingerprint, ID, and SMILES files share
              the same suffix.
            * Data shapes and dtypes are consistent across files.

        Raises:
            AssertionError: If prefixes, shapes, or dtypes are
                inconsistent.
            ValueError: If filenames do not match naming conventions.
        """
        for i in range(self.n_files):
            fp_file = self.fp_files[i]
            id_file = self.id_files[i]
            smi_file = self.smi_files[i]
            fp_name = fp_file.split('/')[-1]
            assert fp_name.startswith(self.fp_prefix)
            id_name = id_file.split('/')[-1]
            assert id_name.startswith(self.id_prefix)
            smi_name = smi_file.split('/')[-1]
            assert smi_name.startswith(self.smi_prefix)
            if fp_name[len(self.fp_prefix):] != id_name[len(self.id_prefix):]:
                raise ValueError(
                    "Problem with library files naming convention: fp file {} does not match id file {}".format(
                        fp_file, id_file))
            if fp_name[len(self.fp_prefix):].split('.')[0] != smi_name[len(self.smi_prefix):].split('.')[0]:
                raise ValueError(
                    "Problem with library files naming convention: fp file {} does not match smi file {}".format(
                        fp_file, smi_file))
            split_fp = fp_name[len(self.fp_prefix):].split('.')
            self.suffixes.append(split_fp[0])  # allows other extensions than .npy (e.g. .npz)
            self.extensions.append(split_fp[1])
            assert id_file.endswith("{}{}.{}".format(self.id_prefix, self.suffixes[-1], self.extensions[-1]))
            # assert smi_file.endswith("{}{}.{}".format(self.smi_prefix, self.suffixes[-1], self.extensions[-1]))
            fps_data = read_np_data(fp_file)
            ids_data = read_np_data(id_file)
            assert fps_data.shape[0] == ids_data.shape[0]
            self.lengths.append(fps_data.shape[0])
            if self.fp_length_bytes is None:
                self.fp_length_bytes = fps_data.shape[1]
            else:
                assert fps_data.shape[1] == self.fp_length_bytes
            assert ids_data.dtype == np.int64
            assert fps_data.dtype == np.uint8
        self.lengths = np.array(self.lengths, dtype=np.int64)
        self.n_mols = sum(self.lengths)

    def load_ids(self, lib_index):
        """Load IDs for a specific library chunk.

        Args:
            lib_index (int): Index of the library file.

        Returns:
            np.ndarray[int64]: Array of molecule IDs.
        Notes:
            This should only be called when ample memory is available. For single-CPU jobs,
            the _process_single_lib_chunked() method from :class:`CSAlgo` is used.
        """
        return np.load(self.id_files[lib_index])

    def load_fps(self, lib_index):
        """Load fingerprints for a specific library chunk.

        Args:
            lib_index (int): Index of the library file.

        Returns:
            np.ndarray[uint8]: Fingerprint array.
        """
        return np.load(self.fp_files[lib_index])

    def load_smiles(self, lib_index):
        """Load SMILES strings for a specific library chunk.

        Args:
            lib_index (int): Index of the library file.

        Returns:
            list[str]: List of SMILES strings.
        """
        with open(self.smi_files[lib_index]) as f:
            lines = f.readlines()
        return [line.strip()[0] for line in lines]

    def load_smiles_indices(self, lib_index, indices):
        """Load SMILES strings for specific indices in a library chunk.

        Args:
            lib_index (int): Index of the library file.
            indices (iterable[int]): Indices of SMILES to load.

        Returns:
            list[str]: List of SMILES strings.
        """
        with open(self.smi_files[lib_index]) as f:
            lines = f.readlines()
        return [lines[i].split()[0] for i in indices]

    def get_full_index(self, lib_index, array_index):
        """Convert a (library index, array index) pair to a full index.

        Args:
            lib_index (int): Library file index.
            array_index (int): Index within the library file.

        Returns:
            int: Full library index (0-based across all files).
        """
        return _full_index_helper(lib_index, array_index, self.lengths)

    def get_lib_array_indices(self, full_index):
        """Convert a full index to (library index, array index).

        Args:
            full_index (int): Full library index.

        Returns:
            tuple[int, int]: (library index, array index) pair.

        Raises:
            ValueError: If ``full_index`` is out of range.
        """
        assert full_index >= 0
        if full_index >= self.n_mols:
            raise ValueError("get_lib_array_indices() called with full_index={} (only {} mols in library)".format(
                full_index, self.n_mols
            ))
        return _lib_array_indices_helper(full_index, self.lengths)


@njit
def _lib_array_indices_helper(full_index, lengths):
    lib_index = 0
    array_index = full_index
    while array_index >= lengths[lib_index]:
        array_index -= lengths[lib_index]
        lib_index += 1
    return lib_index, array_index


@njit
def _full_index_helper(lib_index, array_index, lengths):
    full_index = 0
    for i in range(lib_index):
        full_index += lengths[i]
    full_index += array_index
    return full_index
