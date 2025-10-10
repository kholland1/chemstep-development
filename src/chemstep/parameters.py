

def read_param_file(param_file):
    ordered_keys = ["seed_indices_file",
                    "seed_scores_file",
                    "hit_pprop",
                    "n_docked_per_round",
                    "max_beacons",
                    "max_n_rounds",
                    "bundle_size",
                    "beacon_diversity_strategy",
                    "building_minutes_per_mol",
                    "docking_job_time"]
    floats = {"hit_pprop","building_minutes_per_mol"}
    ints = {"n_docked_per_round", "max_beacons", "max_n_rounds", "bundle_size"}
    strings = {"seed_indices_file", "seed_scores_file", "beacon_diversity_strategy","docking_job_time"}
    params_dict = dict()
    for key in ordered_keys:
        params_dict[key] = None
    with open(param_file) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        data = [x.strip() for x in line.split(":")]
        if len(data) != 2:
            raise ValueError("Problem with the following line in parameter file {}:\n{}".format(param_file, line))
        key, val = data
        if key not in params_dict:
            raise ValueError("Unrecognized parameter in file {}: {}".format(param_file, key))
        if key in floats:
            params_dict[key] = float(val)
        elif key in ints:
            params_dict[key] = int(val)
        else:
            params_dict[key] = val
    for key in params_dict:
        if params_dict[key] is None:
            # Make beacon_diversity_strategy optional for backward compatibility
            if key == "beacon_diversity_strategy":
                params_dict[key] = "maxdiv"
            elif key == "building_minutes_per_mol":
                params_dict[key] = 3
            elif key == "docking_job_time":
                params_dict[key] = "8:00:00"
            else:
                raise ValueError("Parameter {} not found in file {}".format(key, param_file))
    return CSParams(*[params_dict[key] for key in ordered_keys])


class CSParams:
    """ Class storing a parameter set for a run of the ChemSTEP algorithm.

        Attributes:
            seed_indices_file (str): The filename for the full indices (FpLibrary compatible) of the initial docked set,
                                    as an int64 array in .npy format
            seed_scores_file (str): The filename for the scores of the initial docked set, matching the indices of
                                    seed_indices_file, as a float32 array in .npy format
            hit_pprop (float): The definition of docking hit, in terms of pProportion of the library (0.1% = pProp of 3)
            n_docked_per_round (int): The number of molecules to prioritize, build and dock each round
            max_beacons (int): The maximal number of beacons to use.
            max_n_rounds (int): The maximal number of chaining rounds to perform
            beacon_diversity_strategy (str): Strategy for beacon selection - "maxdiv", "entropy_bits", or "mutual_info"
    """
    def __init__(self, seed_indices_file, seed_scores_file, hit_pprop, n_docked_per_round, max_beacons, max_n_rounds, bundle_size, beacon_diversity_strategy="maxdiv",building_minutes_per_mol=3,docking_job_time="8:00:00"):
        self.seed_indices_file = seed_indices_file
        self.seed_scores_file = seed_scores_file
        self.hit_pprop = hit_pprop
        self.n_docked_per_round = n_docked_per_round
        self.max_beacons = max_beacons
        self.max_n_rounds = max_n_rounds
        self.beacon_diversity_strategy = beacon_diversity_strategy
        self.bundle_size = bundle_size
        self.building_minutes_per_mol=building_minutes_per_mol
        self.docking_job_time=docking_job_time

