import subprocess
import time
import os


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
            "job-name": f"chemstep_{self.round_n}",
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
        start_time = time.time()

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
            "o": f"{self.job_folder}/sge_$TASK_ID.out",
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

    def wait(self, job_id, poll_interval=30, timeout=36000):
        print(f"[INFO] Waiting for SGE job array {job_id} to complete...")
        start_time = time.time()

        while True:
            result = subprocess.run(["qstat"], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("Failed to query SGE job status using qstat.")

            running = any(line.startswith(job_id) for line in result.stdout.strip().splitlines())
            if not running:
                print(f"[INFO] SGE job array {job_id} no longer running.")
                return

            if time.time() - start_time > timeout:
                raise TimeoutError(f"SGE job array {job_id} did not complete within {timeout} seconds.")

            time.sleep(poll_interval)