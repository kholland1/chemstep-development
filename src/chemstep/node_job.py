import os
import pickle
import multiprocessing as mp
from typing import List
from chemstep.search_job import run_from_pickle


class NodeJob:
    """
    A node-level job that runs up to `tasks_per_node` SearchJob pickles in parallel.

    Attributes
    ----------
    job_pickle_paths : list[str]
        Paths to SearchJob pickles handled by this node.
    tasks_per_node : int
        Pool size to use within the node.
    """
    def __init__(self, job_pickle_paths: List[str], tasks_per_node: int = 64):
        self.job_pickle_paths = list(job_pickle_paths)
        self.tasks_per_node = int(tasks_per_node)

    def run_job(self):
        if not self.job_pickle_paths:
            return
        if len(self.job_pickle_paths) == 1:
            run_from_pickle(self.job_pickle_paths[0])
            return

        # Prefer 'fork' when available (lower overhead), fallback otherwise.
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()

        with ctx.Pool(self.tasks_per_node) as pool:
            pool.map(run_from_pickle, self.job_pickle_paths)


def run_nodejob_from_pickle(pickle_path: str):
    """Load a NodeJob from disk and execute it."""
    with open(pickle_path, "rb") as f:
        job: NodeJob = pickle.load(f)
    job.run_job()


if __name__ == "__main__":
    # lightweight CLI: python -m chemstep.node_job --pickle <path>
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True)
    args = ap.parse_args()
    run_nodejob_from_pickle(args.pickle)
