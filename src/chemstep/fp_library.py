import numpy as np
from chemstep.utils import read_np_data
from numba import njit
import pickle


def load_library_from_pickle(fn):
    with open(fn, 'rb') as f:
        obj = pickle.load(f)
        assert isinstance(obj, FpLibrary)
        return obj


class FpLibrary:
    """ Class representing a library of molecules to screen, in chemical similarity fingerprint representation.

        Attributes:
            n_files (int): The number of fingerprint files in the FpLibrary
            fp_files (list): The list of paths to the .npy fingerprint files
            id_files (list): The list of paths to the .npy ID (usually ZINC) files. The IDs themselves are int64.
            fp_prefix (str): The start of every fingerprint filename (rest must be identical between fp and id names)
            id_prefix (str): Same as fp_prefix, for the id files
            suffixes (list): The suffixes (exluding the .npy) of library files, in the same order as the one supplied
            lengths (list): The lengths (number of entries) of each library file
            fp_length_bytes (int): The number of bytes in each fingerprint (must be all the same for a given library)
    """

    def __init__(self, fingerprint_files, id_files, smi_files, outname, fp_prefix="fps", id_prefix="zids",
                 smi_prefix="smi"):
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
            # if not str(smi_data.dtype).startswith('|S'):
            #     raise ValueError("Problem with dtype of SMILES data in " +
            #                      "{} (str representation should start with |S but is {} instead)".format(
            #                          smi_file, smi_data.dtype))
        self.lengths = np.array(self.lengths, dtype=np.int64)
        self.n_mols = sum(self.lengths)

    def load_ids(self, lib_index):
        return np.load(self.id_files[lib_index])

    def load_fps(self, lib_index):
        return np.load(self.fp_files[lib_index])

    def load_smiles(self, lib_index):
        with open(self.smi_files[lib_index]) as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def load_smiles_indices(self, lib_index, indices):
        with open(self.smi_files[lib_index]) as f:
            lines = f.readlines()
        return [lines[i].strip() for i in indices]

    def get_full_index(self, lib_index, array_index):
        return _full_index_helper(lib_index, array_index, self.lengths)

    def get_lib_array_indices(self, full_index):
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
