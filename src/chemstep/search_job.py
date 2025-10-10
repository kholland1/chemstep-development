from chemstep.fp_library import FpLibrary
from chemstep.chaining_log import ChainingLog
from chemstep.fingerprints import get_tanimoto_max_excl
from chemstep.utils import read_np_data, mintd_histogram_stream
import numpy as np
import pickle
import os
from numpy.lib.format import open_memmap
import time
from concurrent.futures import ThreadPoolExecutor


def run_from_pickle(pickle_path):
    """
    Load a SearchJob from disk, run it, mark it completed, and overwrite the pickle.
    """
    with open(pickle_path, 'rb') as f:
        job = pickle.load(f)
    if not isinstance(job, SearchJob):
        raise ValueError(f"Pickle at {pickle_path} is not a SearchJob.")

    job.run_job()

    with open(pickle_path, 'wb') as f:
        pickle.dump(job, f)

class SearchJob:
    def __init__(self, unique_id, beacons_array, round_n, fp_library, chaining_log, library_index,
                 scheduler=None, chunk_size=200_000):
        assert isinstance(fp_library, FpLibrary)
        assert isinstance(chaining_log, ChainingLog)
        self.unique_id = unique_id
        self.beacons = beacons_array
        self.round_n = round_n
        self.lib = fp_library
        self.lib_index = library_index
        self.chaining_log = chaining_log
        self.scheduler = scheduler
        self.chunk_size = int(chunk_size)
        self.completed = False

    def run_local(self):
        self.run_job()

    def run_job(self):
        """Run one shard; optionally print per-chunk timing to stdout (SGE picks it up)."""

        fp_path = self.lib.fp_files[self.lib_index]
        excl_path = self.chaining_log.get_filename(
            self.chaining_log.exclusion_prefix, self.chaining_log.get_suffix(self.lib_index))
        mintd_path = self.chaining_log.get_filename(
            self.chaining_log.mintd_prefix, self.chaining_log.get_suffix(self.lib_index))

        n_mols, fp_len = read_np_data(fp_path).shape

        # memmaps
        fps = open_memmap(fp_path, dtype=np.uint8,  mode='r',  shape=(n_mols, fp_len))
        exclusions = open_memmap(excl_path, dtype=np.uint8, mode='r',   shape=(n_mols,))
        mintds = open_memmap(mintd_path, dtype=np.float32, mode='r+', shape=(n_mols,))

        class _Prefetcher:
            """Function-scoped prefetcher; yields (fp_chunk_u64, excl_bool, mintd_view)."""
            def __init__(self, chunk_size, depth=0, ensure_c=True):
                self._ranges = ((s, min(s+chunk_size, n_mols)) for s in range(0, n_mols, chunk_size))
                self._pool = ThreadPoolExecutor(max_workers=depth+1)
                self._q = []
                self.ensure_c = ensure_c
                for _ in range(depth+1):
                    if not self._submit(): break

            def _prep(self, s, e):
                fp_chunk = np.array(fps[s:e], dtype=np.uint8, copy=True)
                excl_chunk = np.array(exclusions[s:e], dtype=np.uint8, copy=True)

                # uint64 casting / bool conversion
                fp64 = fp_chunk.view(np.uint64).reshape(fp_chunk.shape[0], -1)
                excl_bool  = (excl_chunk != 0)
                mint = mintds[s:e]

                if self.ensure_c:
                    if not fp64.flags.c_contiguous: fp64 = np.ascontiguousarray(fp64)
                    if not excl_bool.flags.c_contiguous:  excl_bool  = np.ascontiguousarray(excl_bool)
                return fp64, excl_bool, mint

            def _submit(self):
                try:
                    s, e = next(self._ranges)
                    self._q.append(self._pool.submit(self._prep, s, e))
                    return True
                except StopIteration:
                    return False

            def __iter__(self): return self
            def __next__(self):
                if not self._q:
                    self._pool.shutdown(wait=True)
                    raise StopIteration
                fut = self._q.pop(0)
                self._submit()
                return fut.result()

        # Convert beacons to uint64 for faster TC stuff
        beacons_u64 = self.beacons.view(np.uint64).reshape(self.beacons.shape[0], -1)

        for chunk_idx, (fp_chunk_u64, excl_bool, mintds_c) in enumerate(_Prefetcher(self.chunk_size)):
            mintd_chunk = 1.0 - get_tanimoto_max_excl(beacons_u64, fp_chunk_u64, excl_bool)
            mintds_c[:] = np.minimum(mintds_c, mintd_chunk)

        mintds.flush()
        distrib = mintd_histogram_stream(mintds, exclusions, chunk_size=self.chunk_size)
        np.save(
            self.chaining_log.get_filename(
                self.chaining_log.mintd_distrib_prefix, self.chaining_log.get_suffix(self.lib_index)
            ),
            distrib
        )

        self.completed = True