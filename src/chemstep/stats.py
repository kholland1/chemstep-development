import numpy as np
import glob
from multiprocessing import Pool
from itertools import repeat
from numba import njit


def get_csch_stats(scores_dir, log_dir, score_thresh, n_proc=64):
    # TODO: implement CSAlgo logging where score_thresh etc. is saved
    complete_data = get_complete_data_csch(scores_dir, log_dir, score_thresh, n_proc=n_proc)
    print(complete_data)
    n = len(complete_data)
    stats = np.zeros((n+1, 2))
    for round_n in range(n):
        stats[round_n+1] = [complete_data[round_n, 2] / complete_data[round_n, 3],
                            complete_data[round_n, 0] / complete_data[round_n, 1]]
    return stats


def write_stats_df(scores_dir, log_dir, score_thresh, outname):
    stats = get_csch_stats(scores_dir, log_dir, score_thresh)
    with open(outname, 'w') as f:
        f.write('prop_docked prop_hits\n')
        for row in stats:
            f.write('{} {}\n'.format(*row))


def get_complete_data_csch(scores_dir, log_dir, score_thresh, n_proc=64):
    scores_fns = sorted(glob.glob('{}/scores_*.npy'.format(scores_dir)))
    excl_fns = sorted(glob.glob('{}/excl_*.npy'.format(log_dir)))
    assert len(scores_fns) == len(excl_fns)
    p = Pool(n_proc)
    results = p.starmap(_get_data_single_file, zip(scores_fns, excl_fns, repeat(score_thresh)))
    print(results)
    n = len(results[0])
    for r in results:
        assert len(r) == n
    complete_data = results[0]
    for i in range(1, len(results)):
        complete_data += results[i]
    return complete_data


def _get_data_single_file(scores_fn, excl_fn, score_thresh):
    scores = np.load(scores_fn)
    exclusions = np.load(excl_fn)
    return _get_n_hits_data(scores, exclusions, score_thresh)


@njit
def _get_n_hits_data(scores, exclusions, score_thresh):
    total_hits = np.sum(scores <= score_thresh)
    n_hits_data = np.zeros((exclusions.shape[1], 4), dtype=np.int64)  # fields are n_hits, total_hits, n_docked, total_mols
    for round_n in range(len(n_hits_data)):
        n_docked = 0
        n_hits = 0
        for i in range(exclusions.shape[0]):
            if exclusions[i, round_n]:
                n_docked += 1
                if scores[i] <= score_thresh:
                    n_hits += 1
        n_hits_data[round_n] = n_hits, total_hits, n_docked, len(exclusions)
    return n_hits_data


