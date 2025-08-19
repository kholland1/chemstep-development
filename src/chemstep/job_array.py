import os
import re
import math
import time
import pickle
import subprocess
from pathlib import Path
from chemstep.node_job import NodeJob


class JobArray:
    """
    Abstract base class for submitting and monitoring job arrays on HPC schedulers.
    """
    def __init__(self, round_n, n_jobs, job_folder, job_prefix="job"):
        self.round_n = round_n
        self.n_jobs = n_jobs
        self.job_folder = job_folder
        self.job_prefix = job_prefix

    def submit(self):
        """Submit the job array to the scheduler. Must return a job ID."""
        raise NotImplementedError

    def wait(self, job_id, poll_interval=30):
        """Block until the job array finishes or fails."""
        raise NotImplementedError


class SlurmJobArray(JobArray):
    def __init__(self, round_n, n_jobs, job_folder, python_exec="python", job_prefix="job", slurm_options=None):
        super().__init__(round_n, n_jobs, job_folder, job_prefix)
        self.python_exec = python_exec
        self.slurm_options = slurm_options or {}

    def submit(self):
        script_path = os.path.join(self.job_folder, f"run_array_round_{self.round_n}.sh")
        default_options = {
            "job-name": f"slurm_{self.round_n}",
            "output": f"{self.job_folder}/%x-%A_%a.out",
            "time": "12:00:00",
            "ntasks": "1",
            "mem": "1GB",
            "nodes": "1",
            "cpus-per-task": "1",
            "account": "rrg-mailhoto",
            "array": f"0-{self.n_jobs - 1}",
        }

        opts = {**default_options, **self.slurm_options}

        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            for key, val in opts.items():
                f.write(f"#SBATCH --{key}={val}\n")
            f.write("\n")
            f.write(f"{self.python_exec} -c \"from chemstep.search_job import run_from_pickle; "
                    f"run_from_pickle('{self.job_folder}/{self.round_n}_' + str($SLURM_ARRAY_TASK_ID) + '.pickle')\"\n")

        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Slurm submission failed:\n{result.stderr}")
        job_id = result.stdout.strip().split()[-1]
        print(f"[INFO] Submitted Slurm job array with ID {job_id}")
        return job_id

    def wait(self, job_id, poll_interval=30):
        print(f"[INFO] Waiting for Slurm array job {job_id} to complete...")
        _wait_slurm(job_id, poll_interval)


class SGEJobArray(JobArray):
    """
    Job array submission and tracking for Sun Grid Engine (SGE).
    """

    def __init__(self, round_n, n_jobs, job_folder, python_exec="python", job_prefix="job", sge_options=None):
        super().__init__(round_n, n_jobs, job_folder, job_prefix)
        self.python_exec = python_exec
        self.sge_options = sge_options or {}

    def submit(self):
        script_path = os.path.join(self.job_folder, f"run_array_round_{self.round_n}.sge.sh")
        array_range = f"1-{self.n_jobs}"  # SGE is 1-indexed

        default_options = {
            "cwd": None,
            "j": "y",  # merge stdout and stderr
            "t": array_range,
            "N": f"chemstep_{self.round_n}",
            "o": f"{self.job_folder}/sge_{self.round_n}_$TASK_ID.out",
        }

        opts = {**default_options, **self.sge_options}

        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            for key, val in opts.items():
                if val is None:
                    f.write(f"#$ -{key}\n")
                else:
                    f.write(f"#$ -{key} {val}\n")
            f.write("\n")
            f.write("TID=$(($SGE_TASK_ID - 1))\n")
            f.write(f"{self.python_exec} -c \"from chemstep.search_job import run_from_pickle; "
                    f"run_from_pickle('{self.job_folder}/{self.round_n}_' + str($TID) + '.pickle')\"\n")

        result = subprocess.run(["qsub", script_path], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SGE submission failed:\n{result.stderr}")

        # Parse job ID: e.g., "Your job-array 123456.1-10:1 ("chemstep") has been submitted"
        job_line = result.stdout.strip()
        job_id = job_line.split()[2].split(".")[0]
        print(f"[INFO] Submitted SGE job array with ID {job_id}")
        return job_id

    def wait(self, job_id, poll_interval=30):
        """
        Block until every task in the SGE array has completed.

        Strategy
        --------
        1. Poll qstat -u $USER          → job still queued / running?
        2. When it disappears, poll qacct -j JOBID -t
           • Records exist → finished  (check failed/exit_status)
           • Records absent → accounting lag; keep waiting
        """
        print(f"[INFO] Waiting for SGE job array {job_id} to complete...")
        t0 = time.time()

        while True:
            # 1) still in the queue?
            qstat = subprocess.run(
                ["qstat", "-u", os.getenv("USER")],
                capture_output=True, text=True
            )
            in_qstat = any(
                line.split()[0].startswith(job_id)
                for line in qstat.stdout.strip().splitlines()
            )

            if in_qstat:
                # queued (qw) or running (r/R/t)
                pass
            else:
                # 2) check accounting; -t lists every task record
                acc = subprocess.run(
                    ["qacct", "-j", job_id, "-t"],
                    capture_output=True, text=True
                )

                if acc.returncode == 0 and acc.stdout.strip():
                    # finished – look for failed tasks
                    if "failed" in acc.stdout.lower():
                        raise RuntimeError(
                            f"Some tasks in array {job_id} reported failure:\n{acc.stdout}"
                        )
                print(f"[INFO] SGE job array {job_id} appears to have completed.")
                return

            time.sleep(poll_interval)


def _render_slurm_options(opts):
    """
    Render arbitrary SBATCH options from a dict into header lines.
    - Boolean True => '#SBATCH --key'
    - False/None/'' => skipped
    - other          => '#SBATCH --key=value'
    Keys may be provided with or without leading dashes.
    """
    lines = []
    if not opts:
        return lines
    for k, v in opts.items():
        if k.startswith("--") or k.startswith("-"):
            key = k
        else:
            key = f"--{k}"

        if v is True:
            lines.append(f"#SBATCH {key}")
        elif v in (False, None, ""):
            continue
        else:
            lines.append(f"#SBATCH {key}={v}")
    return lines


class SlurmNodeArray:
    """
    Launch a Slurm array where *each array task reserves a full node* and executes
    a single **NodeJob pickle** (wrapping up to `tasks_per_node` SearchJob pickles).
    Inside the node, chemstep.node_job.NodeJob spawns a local Pool(tasks_per_node)
    and runs its SearchJobs in parallel.

    This class does not discover files. It deterministically constructs
    SearchJob pickle paths as:
        f"{jobs_folder}/{round_n}_{i}.pickle" for i in range(n_jobs)

    Parameters
    ----------
    jobs_folder : str
        Folder containing the SearchJob pickles.
    round_n : int | str
        Round identifier used in pickle naming.
    n_jobs : int
        Number of SearchJob pickles (indexed [0, n_jobs)).
    tasks_per_node : int, default 64
        Number of SearchJobs to run concurrently per node (Pool size and CPU request).
    python_exec : str, default "python"
        Python interpreter to invoke in the Slurm script.
    slurm_options : dict, optional
        Extra SBATCH options (e.g., {"account":"rrg-foo", "partition":"cpu", "exclusive":True}).
        Conflicting options are ignored (see _SBATCH_RESERVED).
    job_name : str, optional
        Slurm job name; defaults to "chemstep_r{round}_node".
    time_limit : str, optional
        Walltime (HH:MM:SS). If omitted, will use slurm_options['time'] if present,
        otherwise defaults to "08:00:00".
    """

    # SBATCH fields that we always set ourselves—skip them if provided in slurm_options
    _SBATCH_RESERVED = {
        "array", "nodes", "ntasks", "cpus-per-task", "cpus", "job-name", "J",
        "output", "error", "o", "e", "time", "mem",
    }

    def __init__(self, jobs_folder, round_n, n_jobs, tasks_per_node=64, python_exec="python", slurm_options=None,
                 job_name=None, time_limit=None):
        if n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")
        if tasks_per_node <= 1:
            raise ValueError("tasks_per_node must be > 1")

        self.jobs_folder = str(jobs_folder)
        self.round_n = round_n
        self.n_jobs = int(n_jobs)
        self.tasks_per_node = int(tasks_per_node)
        self.python_exec = python_exec
        self.slurm_options = dict(slurm_options or {})
        self.job_name = job_name or f"chemstep_r{round_n}_node"

        # time handling
        if time_limit is None and "time" in self.slurm_options:
            self.time_limit = str(self.slurm_options.pop("time"))
        else:
            self.time_limit = time_limit or "01:00:00"

        # Derive array size (still write scripts even if n_jobs == 0 for consistency)
        self.n_nodes = max(0, math.ceil(self.n_jobs / self.tasks_per_node))
        assert self.n_nodes > 0

        # Paths
        pdir = Path(self.jobs_folder)
        pdir.mkdir(parents=True, exist_ok=True)
        self.script_path = str(pdir / f"slurm_node_array_round_{self.round_n}.sh")
        self.stdout_tmpl = str(pdir / "slurm-node-%A_%a.out")
        self.stderr_tmpl = str(pdir / "slurm-node-%A_%a.err")

        # NodeJob pickle prefix (one per array index)
        self.node_pickle_prefix = str(pdir / f"{self.round_n}_node")
        self.job_id = None
        self._write_script()
        self._prepare_node_pickles()

    def wait(self, poll_interval=30):
        """Block until the Slurm array completes."""
        if not self.job_id:
            raise RuntimeError("wait() called before submit()")
        job_id = self.job_id
        print(f"[INFO] Waiting for Slurm node array job {job_id} to complete...")
        _wait_slurm(job_id, poll_interval)

    def _get_pickle_path(self, i: int) -> str:
        """jobs_folder / f\"{round_n}_{i}.pickle\""""
        return os.path.join(self.jobs_folder, f"{self.round_n}_{i}.pickle")

    def _prepare_node_pickles(self):
        """
        Chunk the SearchJob pickles into blocks of size `tasks_per_node`
        and write one NodeJob pickle per block at:
            {jobs_folder}/{round_n}_node_{array_index}.pickle
        """
        for a in range(self.n_nodes):
            start = a * self.tasks_per_node
            end = min(start + self.tasks_per_node, self.n_jobs)
            chunk_paths = [self._get_pickle_path(i) for i in range(start, end)]
            node_pickle_path = f"{self.node_pickle_prefix}_{a}.pickle"
            with open(node_pickle_path, "wb") as fh:
                pickle.dump(NodeJob(chunk_paths, tasks_per_node=self.tasks_per_node), fh)

    def _write_script(self) -> None:
        """Write the Slurm array script that loads and runs a NodeJob pickle."""
        array_spec = f"0-{self.n_nodes - 1}"
        header = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --array={array_spec}",
            "#SBATCH --nodes=1",
            "#SBATCH --mem=0",
            f"#SBATCH --ntasks={self.tasks_per_node}",
            f"#SBATCH --cpus-per-task=1",
            f"#SBATCH --time={self.time_limit}",
            f"#SBATCH --output={self.stdout_tmpl}",
            f"#SBATCH --error={self.stderr_tmpl}",
        ]

        # Merge extra SBATCH options, skipping any we already set
        extra = {}
        for k, v in self.slurm_options.items():
            kk = k.lstrip("-")
            if kk in self._SBATCH_RESERVED:
                continue
            extra[k] = v
        header.extend(_render_slurm_options(extra))

        with open(self.script_path, "w") as f:
            f.write("\n".join(header) + "\n")
            f.write(f"{self.python_exec} -c \"from chemstep.node_job import run_nodejob_from_pickle; "
                    f"run_nodejob_from_pickle('{self.node_pickle_prefix}_' + str($SLURM_ARRAY_TASK_ID) + '.pickle')\"\n")

    def submit(self) -> str:
        """Submit the array script and record the Slurm job ID."""
        proc = subprocess.run(["sbatch", self.script_path], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}\n(while submitting {self.script_path})")
        m = re.search(r"Submitted batch job (\d+)", proc.stdout)
        if not m:
            raise RuntimeError(f"Unable to parse job id from sbatch output: {proc.stdout}")
        self.job_id = m.group(1)
        print(
            f"[INFO] Submitted Slurm *node* array {self.job_id} with {self.n_nodes} tasks; "
            f"{self.tasks_per_node} SearchJobs per node (total {self.n_jobs})."
        )
        return self.job_id


def _wait_slurm(job_id, poll_interval):
    while True:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobID,State", "--parsable2", "--noheader"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        task_states = [
            line.split("|")[1]
            for line in lines
            if "_" in line and job_id in line
        ]

        if not task_states:
            print("[INFO] Waiting for Slurm to register array tasks...")
        elif all(state == "COMPLETED" for state in task_states):
            print("[INFO] All Slurm array tasks completed.")
            return
        elif any(state in {"FAILED", "CANCELLED", "TIMEOUT"} for state in task_states):
            raise RuntimeError(f"Slurm job array {job_id} has failed tasks: {task_states}")

        time.sleep(poll_interval)
