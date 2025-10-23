'''
This is to do automatic building and docking for chemstep
NOTES:
There is lots of hardcoded shit in here. It will only work for the 13B space afaik.
'''

from chemstep.docking_algorithm import DockingAlgorithm
from chemstep.fp_library import FpLibrary
import numpy as np
from numpy.lib.format import open_memmap

import sys
sys.path.append('/wynton/group/bks/soft/DOCK-3.8.5/DOCK3.8/zinc22-3d/submit')
from submit_building_docker import make_building_array_job
import subprocess
import os
from pathlib import Path
import glob
import time
import redis

from chemstep.id_helper import char_to_int64

class AutoDocking(DockingAlgorithm):

    def __init__(self, lib_array_indices, smi_list, fp_lib, smi_file_path, round_n, dockfiles_path, algo_params, verbose=False):
        super().__init__(lib_array_indices, smi_list)
        assert isinstance(fp_lib, FpLibrary)
        self.lib_arr_indices = lib_array_indices
        self.verbose = verbose
        self.smi_file_path = smi_file_path
        self.round_n = round_n
        self.dockfiles_path = dockfiles_path
        self.bundle_size = algo_params.bundle_size
        self.building_minutes_per_mol = float(getattr(algo_params, "builing_minutes_per_mol", 3))
        self.docking_job_time = getattr(algo_params, "docking_job_time", "8:00:00")

    def dock_all(self, indices_skipped=None, redis_db_host=None, redis_db_port=None, redis_password="chemstep"):  # TODO: make parallel (low priority)

        cwd = Path.cwd()
        building_dir = (cwd / f"round_{self.round_n}_building").resolve()
        docking_dir = (cwd / f"round_{self.round_n}_docking").resolve()
        bundle_paths_file = (cwd / f"round_{self.round_n}_building" / "bundle_paths.sdi").resolve()
        dockfiles_path = Path(self.dockfiles_path).resolve()

        # 1. Run find to generate bundle_paths.sdi
        with open(bundle_paths_file, "w") as f:
            subprocess.run(
                ["find", str(building_dir), "-type", "f", "-name", "bundle.db2.tgz"],
                stdout=f,
                check=True
            )

        # 2. Build environment variables
        env = os.environ.copy()
        env["MOLECULES_DIR_TO_BIND"] = str(building_dir)
        env["DOCKFILES"] = str(dockfiles_path)
        env["INPUT_FOLDER"] = str(building_dir)
        env["OUTPUT_FOLDER"] = str(docking_dir)
        env['USE_SGE_ARGS'] = f"-l h_rt={self.docking_job_time}"

        # Submit and wait for docking jobs
        result = subprocess.run(
            ["/wynton/group/bks/work/bwhall61/needs_github/super_dock3r.sh"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            check=True
        )
        last_line = result.stdout.strip().splitlines()[-1]
        job_id = last_line.split()[2].split(".")[0]
        self.wait_sge(job_id)

        # Get any precomputed scores
        precomputed_scores = {}
        if redis_db_host and redis_db_port:
            r = redis.Redis(host=redis_db_host, port=redis_db_port, password=redis_password, decode_responses=True) 
            try:
                r.ping()
            except:
                raise ValueError("Redis server does not seem responsive")
            for full_index in indices_skipped:
                zid, score = r.hmget(full_index, "zid", "score")
                precomputed_scores[zid] = float(score)

        # Get any new score file and combine into the scores_round_*.txt
        outdock_filenames = glob.glob(f'{env["OUTPUT_FOLDER"]}/*/*/OUTDOCK.*')
        self.write_scores_df(outdock_filenames, f'scores_round_{self.round_n}.txt', precomputed_scores)

        # Convert to scores and indices npy. 
        with open(f'scores_round_{self.round_n}.txt', 'r') as f:
            lines = f.readlines()

        scores = np.zeros(len(lines), dtype=np.float32)
        indices = np.zeros(len(lines), dtype=np.int64)

        for i, line in enumerate(lines):
            lib_id, score = line.strip().split()
            lib_id_trunc = lib_id[3:]
            full_index = char_to_int64(lib_id_trunc)
            score = float(score)
            # Insert scores into the database
            if redis_db_host and redis_db_port and score != 100 and full_index not in indices_skipped:
                r.hset(full_index, mappping={"zid": lib_id, "score": score})
            scores[i] = score
            indices[i] = full_index

        np.save(f'scores_round_{self.round_n}.npy', scores)
        np.save(f'indices_round_{self.round_n}.npy', indices)
        self.scores_list = scores
        return indices

        # Probably do some sort of assert to do checking

    def build_all(self):
        make_building_array_job(
            self.smi_file_path, # input smi file
            f"round_{self.round_n}_building", # output folder
            self.bundle_size, # bundle size
            self.building_minutes_per_mol, # minutes per mol
            None, # building_config_file. Not implemented yet
            f"round_{self.round_n}_building.sh", # array_job_name
            True, # skip_name_check
            'sge', # scheduler
            'apptainer', # container_software
            '/wynton/group/bks/work/bwhall61/CHEMSTEP_MEGA_FIXING_FOLDER/building_no_strain.sif'
            # '/wynton/group/bks/soft/DOCK-3.8.5/building_pipeline.sif'# container path or name
        )

        # submit building
        result = subprocess.run(
            ["qsub", f"round_{self.round_n}_building.sh"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        output=result.stdout.strip()
        print(f'qsub output: {output}')

        job_id = output.split()[2].split('.')[0]
        self.wait_sge(job_id)

        

    def get_scores_list(self):
        return self.scores_list

    def wait_sge(self, job_id, poll_interval=30):
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

    def get_outdock_score_dict(self, outdock_fn, mol_id_prefix='MOL', undocked_score=100):
        with open(outdock_fn) as f:
            lines = f.readlines()
        i = 0
        data_dict = dict()
        while i < len(lines):
            line = lines[i]
            ll = line.split()
            if len(ll) >= 2 and ll[1].startswith(mol_id_prefix):
                zincid = ll[1]
                try:
                    score = float(ll[-1])
                except ValueError:
                    score = undocked_score
                if zincid in data_dict:
                    if score < data_dict[zincid]:
                        data_dict[zincid] = score
                else:
                    data_dict[zincid] = score
            i += 1
        return data_dict


    def write_scores_df(self, outdocks_list, outname, precomputed_scores={}):
        dd = dict()
        for od in outdocks_list:
            d = self.get_outdock_score_dict(od)
            self.fuse_data_dicts(dd, d)

        # Add any precomputed scores from the db
        self.fuse_data_dicts(dd, precomputed_scores)

        with open(outname, 'w') as f:
            for zid in dd:
                f.write('{} {}\n'.format(zid, dd[zid]))


    def fuse_data_dicts(self, ref_dict, add_dict):
        for key in add_dict:
            if key not in ref_dict:
                ref_dict[key] = add_dict[key]
            else:
                newval = add_dict[key]
                if newval < ref_dict[key]:
                    ref_dict[key] = newval
