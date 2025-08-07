from abc import ABCMeta, abstractmethod


class DockingAlgorithm(metaclass=ABCMeta):

    def __init__(self, lib_array_indices, smi_list):
        self.lib_array_indices = lib_array_indices
        self.smi_list = smi_list
        self.scores_list = None

    @abstractmethod
    def dock_all(self):
        pass

    @abstractmethod
    def get_scores_list(self):
        pass
