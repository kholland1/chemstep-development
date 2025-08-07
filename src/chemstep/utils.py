import numpy as np


class NDAData:
    __slots__ = ['shape', 'fortran', 'dtype', 'minor', 'major']

    def __init__(self, major, minor, shape, fortran, dtype):
        self.major = major
        self.minor = minor
        self.shape = shape
        self.fortran = fortran
        self.dtype = dtype


def read_np_data(fn):
    with open(fn, 'rb') as f:
        major, minor = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    return NDAData(major, minor, shape, fortran, dtype)


def mintd_histogram_stream(mintds, exclusions, chunk_size=1_000_000):
    """
    Return the 1001-bin histogram used by ChemSTEP, **excluding** molecules
    whose exclusion flag is already >0.
    """
    distrib = np.zeros(1001, dtype=np.int32)
    n = mintds.shape[0]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        md = mintds[start:end]
        excls = exclusions[start:end]

        # keep only still-eligible molecules with mintd ≤ 1
        mask = (excls == 0) & (md <= 1.0)
        if not mask.any():
            continue

        bins = np.round(md[mask] * 1000).astype(np.int64)
        np.add.at(distrib, bins, 1)

    return distrib
