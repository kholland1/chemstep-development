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

    # job.profile=True
    job.run_job()

    with open(pickle_path, 'wb') as f:
        pickle.dump(job, f)

class SearchJob:
    def __init__(self, unique_id, beacons_array, round_n, fp_library, chaining_log, library_index,
                 scheduler=None, chunk_size=200_000, profile=False, profile_every=1, show_density=False):
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

        # minimal profiling toggles
        self.profile = bool(profile)
        self.profile_every = max(1, int(profile_every))   # print every N chunks
        self.show_density = bool(show_density)            # compute cheap density proxy

    def run_local(self):
        self.run_job()

    def run_job(self):
        """Run one shard; optionally print per-chunk timing to stdout (SGE picks it up)."""
        t_job0 = time.perf_counter()

        fp_path = self.lib.fp_files[self.lib_index]
        excl_path = self.chaining_log.get_filename(
            self.chaining_log.exclusion_prefix, self.chaining_log.get_suffix(self.lib_index))
        mintd_path = self.chaining_log.get_filename(
            self.chaining_log.mintd_prefix, self.chaining_log.get_suffix(self.lib_index))

        # discover shape (fast)
        n_mols, fp_len = read_np_data(fp_path).shape

        # memmaps
        t_open0 = time.perf_counter()
        fps = open_memmap(fp_path, dtype=np.uint8,  mode='r',  shape=(n_mols, fp_len))
        exclusions = open_memmap(excl_path, dtype=np.uint8, mode='r',   shape=(n_mols,))
        mintds = open_memmap(mintd_path, dtype=np.float32, mode='r+', shape=(n_mols,))
        t_open1 = time.perf_counter()

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

                # isolate I/O/slice time
                t0 = time.perf_counter()
                # fp_chunk = fps[s:e]
                # excl_chunk = exclusions[s:e]
                fp_chunk = np.array(fps[s:e], dtype=np.uint8, copy=True)
                excl_chunk = np.array(exclusions[s:e], dtype=np.uint8, copy=True)
                t1 = time.perf_counter() 

                # uint64 casting / bool conversion
                t6 = time.perf_counter()
                fp64 = fp_chunk.view(np.uint64).reshape(fp_chunk.shape[0], -1)
                excl_bool  = (excl_chunk != 0)
                t7 = time.perf_counter()

                # mintd slice time
                t4 = time.perf_counter()
                mint = mintds[s:e]
                t5 = time.perf_counter()

                if self.ensure_c:
                    if not fp64.flags.c_contiguous: fp64 = np.ascontiguousarray(fp64)
                    if not excl_bool.flags.c_contiguous:  excl_bool  = np.ascontiguousarray(excl_bool)
                return fp64, excl_bool, mint, (t1-t0), (t7-t6), (t5-t4), s, e

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


        # print header once
        if self.profile:
            print(
                f"# PROF header ts={time.strftime('%Y-%m-%dT%H:%M:%S')} "
                f"job={self.unique_id} round={self.round_n} lib={self.lib_index} "
                f"n_mols={n_mols} fp_len={fp_len} open_s={t_open1 - t_open0:.4f}",
                flush=True
            )
            print("# chunk  start   end     excl0   excl_rate  copy_s  kernel_s  write_s  cast_s  chunk_s"
                  + ("  mean_byte" if self.show_density else ""),
                  flush=True)

        total_copy = total_kernel = total_write = total_cast = 0.0
        chunks = 0


        # Convert beacons to uint64 for faster TC stuff
        beacons_u64 = self.beacons.view(np.uint64).reshape(self.beacons.shape[0], -1)

        # for chunk_idx, start in enumerate(range(0, n_mols, self.chunk_size)):
        for chunk_idx, (fp_chunk_u64, excl_bool, mintds_c, t_io, t_cast, tmint_io, start, end) in enumerate(_Prefetcher(self.chunk_size)):

            # end = min(start + self.chunk_size, n_mols) #COMMENT OUT BC PREFETCH
            t_chunk0 = time.perf_counter()

            # isolate I/O/copy time
            # t0 = time.perf_counter() #COMMENT OUT BC PREFETCH
            # fp_chunk = np.array(fps[start:end], dtype=np.uint8, copy=True) #COMMENT OUT BC PREFETCH
            # excl_chunk = np.array(exclusions[start:end], dtype=np.uint8, copy=True) #COMMENT OUT BC PREFETCH
            # t1 = time.perf_counter() #COMMENT OUT BC PREFETCH

            # uint64 Casting
            # t6 = time.perf_counter() #COMMENT OUT BC PREFETCH
            # fp_chunk_u64 = fp_chunk.view(np.uint64).reshape(fp_chunk.shape[0], -1) #COMMENT OUT BC PREFETCH
            # excl_bool    = (excl_chunk != 0) #COMMENT OUT BC PREFETCH
            # t7 = time.perf_counter() #COMMENT OUT BC PREFETCH

            # kernel
            t2 = time.perf_counter()
            tani = get_tanimoto_max_excl(beacons_u64, fp_chunk_u64, excl_bool)  # current kernel
            mintd_chunk = 1.0 - tani
            t3 = time.perf_counter()

            # write-back
            t4 = time.perf_counter()
            # mintds[start:end] = np.minimum(mintds[start:end], mintd_chunk) #COMMENT OUT BC PREFETCH
            mintds_c = np.minimum(mintds_c, mintd_chunk)
            t5 = time.perf_counter()

            # copy_s   = t1 - t0 #COMMENT OUT BC PREFETCH
            copy_s = t_io
            kernel_s = t3 - t2
            write_s  = (t5 - t4) + tmint_io
            # cast_s   = t7 - t6 #COMMENT OUT BC PREFETCH
            cast_s = t_cast
            chunk_s  = time.perf_counter() - t_chunk0

            total_copy   += copy_s
            total_kernel += kernel_s
            total_write  += write_s
            total_cast += cast_s
            chunks += 1

            if self.profile and (chunk_idx % self.profile_every == 0):
                # non_excl = int((excl_chunk == 0).sum()) #COMMENT OUT BC PREFETCH
                non_excl = int((~excl_bool).sum())
                total = end - start
                excl_rate = 1.0 - (non_excl / total if total else 0.0)
                if self.show_density and total > 0:
                    # mean_byte = float(fp_chunk.mean())  #COMMENT OUT BC PREFETCH
                    mean_byte = float(fp64.view(np.uint8).mean())
                    print(f"{chunk_idx:6d} {start:7d} {end:7d} {non_excl:7d} {excl_rate:9.6f} "
                          f"{copy_s:7.4f} {kernel_s:9.4f} {write_s:8.4f} {cast_s:8.4f} {chunk_s:8.4f} {mean_byte:9.2f}",
                          flush=True)
                else:
                    print(f"{chunk_idx:6d} {start:7d} {end:7d} {non_excl:7d} {excl_rate:9.6f} "
                          f"{copy_s:7.4f} {kernel_s:9.4f} {write_s:8.4f} {cast_s:8.4f} {chunk_s:8.4f}",
                          flush=True)

        # flush and histogram timings
        t_flush0 = time.perf_counter()
        mintds.flush()
        t_flush1 = time.perf_counter()

        t_hist0 = time.perf_counter()
        distrib = mintd_histogram_stream(mintds, exclusions, chunk_size=self.chunk_size)
        np.save(
            self.chaining_log.get_filename(
                self.chaining_log.mintd_distrib_prefix, self.chaining_log.get_suffix(self.lib_index)
            ),
            distrib
        )
        t_hist1 = time.perf_counter()

        total_s = time.perf_counter() - t_job0
        if self.profile:
            print(
                f"# PROF summary job={self.unique_id} round={self.round_n} lib={self.lib_index} "
                f"chunks={chunks} total_copy_s={total_copy:.3f} total_kernel_s={total_kernel:.3f} "
                f"total_write_s={total_write:.3f} total_cast_s={total_cast:.3f} flush_s={t_flush1 - t_flush0:.3f} "
                f"hist_s={t_hist1 - t_hist0:.3f} total_job_s={total_s:.3f}",
                flush=True
            )

        self.completed = True