

def read_param_file(param_file):
    ordered_keys = ["seed_scores_file",
                    "novelty_set_file",
                    "novelty_dist_thresh",
                    "screen_novelty",
                    "beacon_dist_thresh",
                    "diversity_dist_thresh",
                    "hit_pprop",
                    "artefact_pprop",
                    "use_artefact_filter",
                    "n_docked_per_round",
                    "max_beacons",
                    "max_n_rounds"]
    floats = {"novelty_dist_thresh", "diversity_dist_thresh", "beacon_dist_thresh", "hit_pprop", "artefact_pprop"}
    ints = {"n_docked_per_round", "max_beacons", "max_n_rounds"}
    bools = {"screen_novelty", "use_artefact_filter"}
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
        elif key in bools:
            params_dict[key] = bool(eval(val))
        else:
            params_dict[key] = val
    for key in params_dict:
        if params_dict[key] is None:
            raise ValueError("Parameter {} not found in file {}".format(key, param_file))
    return CSParams(*[params_dict[key] for key in ordered_keys])


class CSParams:
    """ Class storing a parameter set for a run of the ChemSTEP algorithm.

        Attributes:
            seed_scores_file (str): The filename for the initial set of scored (docked) molecules, as either a
                                    dictionary or efficient hash table of int64 -> float64 (in pickle format)
            novelty_set_file (str): A .npy file containing all fingerprints of known binders, used for eliminating
                                      molecules below the diversity distance threshold
            novelty_dist_thresh (float): The novelty distance threshold, in Jaccard (Tanimoto) ***distance***, NOT
                                         similarity!
            screen_novelty (bool): If True, all ligands in the database closer than the novelty_dist_thresh to the
                                   novelty set will be excluded before the first round.
            beacon_dist_thresh (float): The distance threshold for consideration of a molecule as a beacon of a
                                        subsequent rount. This is what ensures "chemical space travel". Again, expressed
                                        as a Jaccard distance.
            diversity_dist_thresh (float): The diversity distance threshold (for beacon selection), in Jaccard
                                           (Tanimoto) distance, NOT similarity!
            hit_pprop (float): The definition of docking hit, in terms of pProportion of the library (0.1% = pProp of 3)
            artefact_pprop (float): Same as hit_pprop, but for where one expects artefacts to dominate.
            use_artefact_filter (bool): If True, an artefact threshold score will be set. Scores lower than the artefact
                                        threshold get "reversed", that is their score for beacon selection purposes are
                                        set to: new_score = artefact_threshold + (artefact_threshold - old_score)
            n_docked_per_round (int): The number of molecules to prioritize, build and dock each round
            max_beacons (int): The maximal number of beacons to use.
            max_n_rounds (int): The maximal number of chaining rounds to perform
    """
    def __init__(self, seed_scores_file, novelty_set_file, novelty_dist_thresh, screen_novelty, beacon_dist_thresh,
                 diversity_dist_thresh, hit_pprop, artefact_pprop, use_artefact_filter, n_docked_per_round, max_beacons,
                 max_n_rounds):
        self.seed_scores_file = seed_scores_file
        self.novelty_set_file = novelty_set_file
        self.novelty_dist_thresh = novelty_dist_thresh
        self.screen_novelty = screen_novelty
        self.diversity_dist_thresh = diversity_dist_thresh
        self.beacon_dist_thresh = beacon_dist_thresh
        self.hit_pprop = hit_pprop
        self.artefact_pprop = artefact_pprop
        self.use_artefact_filter = use_artefact_filter
        self.n_docked_per_round = n_docked_per_round
        self.max_beacons = max_beacons
        self.max_n_rounds = max_n_rounds

