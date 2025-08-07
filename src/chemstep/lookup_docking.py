from chemstep.docking_algorithm import DockingAlgorithm
from chemstep.fp_library import FpLibrary
import numpy as np


class LookupDocking(DockingAlgorithm):

    def __init__(self, lib_array_indices, smi_list, scores_fns, fp_lib, verbose=False):
        super().__init__(lib_array_indices, smi_list)
        assert isinstance(fp_lib, FpLibrary)
        assert len(scores_fns) == fp_lib.n_files
        self.scores_fns = scores_fns
        self.lib_arr_indices = lib_array_indices
        self.verbose = verbose

    def dock_all(self):  # TODO: make parallel (low priority)
        self.scores_list = np.zeros(len(self.lib_arr_indices), dtype=np.float32)
        count = 0
        last_lib_index = None
        i = 0
        while i < len(self.lib_arr_indices):
            lib_index, array_index = self.lib_arr_indices[i]
            if lib_index != last_lib_index:
                indices = []
                while i < len(self.lib_arr_indices) and self.lib_arr_indices[i][0] == lib_index:
                    indices.append(self.lib_arr_indices[i][1])
                    i += 1
                self.scores_list[count:count+len(indices)] = np.load(self.scores_fns[lib_index], mmap_mode='r')[indices]
                count += len(indices)

                if self.verbose:
                    print("Now docking lib_index {}".format(lib_index))
            else:
                i += 1
            last_lib_index = lib_index
        assert count == len(self.scores_list)

    def get_scores_list(self):
        return self.scores_list
