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
