from numba import njit, prange
import numpy as np
import sys


@njit
def get_tanimoto_max(query_fps, database_fps):
    results = np.zeros(len(database_fps))
    for i in range(len(database_fps)):
        dbfp = database_fps[i]
        results[i] = update_results_maxtani(dbfp, query_fps)
    return results


@njit(parallel=True)
def get_tanimoto_max_excl(query_fps, database_fps, excl):
    n = len(database_fps)
    results = np.full(n, -1.0)
    for i in prange(n):
        if not excl[i]:
            results[i] = update_results_maxtani(database_fps[i], query_fps)
    return results


@njit()
def update_results_maxtani(dbfp, query_fps):
    tmp = np.zeros(len(query_fps))
    for j in range(len(query_fps)):
        tmp[j] = get_tc(dbfp, query_fps[j])
    return np.max(tmp)

# New function that also returns which beacon was the maxTC (minTD)
@njit(parallel=True)
def get_tanimoto_max_excl_idx(query_fps, database_fps, excl):
    n = len(database_fps)
    max_tc = np.full(n, -1.0)
    arg_idx = np.full(n, -1)  # -1 = no result (excluded)
    for i in prange(n):
        if not excl[i]:
            best = -1.0
            best_j = -1
            for j in range(len(query_fps)):
                tc = get_tc(database_fps[i], query_fps[j])
                if tc > best:
                    best = tc
                    best_j = j
            max_tc[i] = best
            arg_idx[i] = best_j
    return max_tc, arg_idx


@njit(inline='always')
def popcnt64(x):
    x = x - ((x >> 1) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
    x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return x

@njit(fastmath=True, inline='always')
def get_tc(fp1_u64, fp2_u64):
    inter = 0
    a_sum = 0
    b_sum = 0
    for i in range(fp1_u64.size):
        A = fp1_u64[i]; B = fp2_u64[i]
        a_sum += int(popcnt64(A))
        b_sum += int(popcnt64(B))
        inter += int(popcnt64(A & B))
    denom = a_sum + b_sum - inter
    return 0.0 if denom == 0 else inter / float(denom)



def get_fp_from_smiles(smiles, n_bits=1024, packbits=True):
    if "Chem" not in sys.modules:
        from rdkit import Chem
    if "AllChem" not in sys.modules:
        from rdkit.Chem import AllChem
    info = {}
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=False, radius=2, nBits=n_bits, bitInfo=info)
    fp = np.array(fp)
    if packbits:
        fp = np.packbits(fp)
    return fp


def compute_morgan_fps(smi_fn, n_bits=1024, ignore_exceptions=False):
    assert n_bits % 8 == 0
    with open(smi_fn) as f:
        lines = f.readlines()
    all_fps = np.zeros((len(lines), int(n_bits/8)), dtype=np.uint8)
    for i, line in enumerate(lines):
        smiles = line.split()[0]
        try:
            fp = get_fp_from_smiles(smiles, n_bits=n_bits, packbits=True)
            all_fps[i] = fp
        except Exception as e:
            if ignore_exceptions:
                print(f"Ignoring exception for line {i} of file {smi_fn} ( SMILES: {smiles} )")
                continue
            else:
                raise e
    return all_fps

