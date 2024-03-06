from csch.fp_library import FpLibrary
from csch.chaining_log import ChainingLog
from bksltk.fingerprints import get_tanimoto_max_excl


class SearchJob:
    """ Class representing a search job across a subset of the library, from a list of beacons. Can either run locally
        or generate a Slurm/other job that gets submitted to a scheduler.
    """

    def __init__(self, unique_id, beacons_array, round_n, fp_library, chaining_log, library_index, scheduler=None):
        assert isinstance(fp_library, FpLibrary)
        assert isinstance(chaining_log, ChainingLog)
        self.unique_id = unique_id
        self.beacons = beacons_array
        self.round_n = round_n
        self.lib = fp_library
        self.lib_index = library_index
        self.chaining_log = chaining_log
        self.scheduler = scheduler

    def run_local(self):
        database_array = self.lib.load_fps(self.lib_index)
        exclusions = self.chaining_log.load_exclusions(self.lib_index, self.round_n - 1)
        mintds = 1 - get_tanimoto_max_excl(self.beacons, database_array, exclusions)
        self.chaining_log.add_mintds(mintds, self.lib_index, self.round_n)

