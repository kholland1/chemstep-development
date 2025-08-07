from numba import njit
import numpy as np
import sys


bitsum_lookup = np.zeros(256, dtype=np.float32)
for ix in range(256):
    bitsum_lookup[ix] = np.sum(np.unpackbits(np.array([ix], dtype=np.uint8)))


@njit
def get_tanimoto_max(query_fps, database_fps):
    results = np.zeros(len(database_fps))
    for i in range(len(database_fps)):
        dbfp = database_fps[i]
        results[i] = update_results_maxtani(dbfp, query_fps)
    return results


@njit
def get_tanimoto_max_excl(query_fps, database_fps, excl):
    results = np.full(len(database_fps), -1.0)
    for i in range(len(database_fps)):
        if excl[i]:
            continue
        dbfp = database_fps[i]
        results[i] = update_results_maxtani(dbfp, query_fps)
    return results


@njit
def update_results_maxtani(dbfp, query_fps):
    tmp = np.zeros(len(query_fps))
    for j in range(len(query_fps)):
        qfp = query_fps[j]
        tmp[j] = get_tc(dbfp, qfp)
    return np.max(tmp)


@njit
def get_tc(fp1, fp2):
    denom = bitsum(np.bitwise_or(fp1, fp2))
    if denom == 0:
        return 0.0
    else:
        return bitsum(np.bitwise_and(fp1, fp2)) / denom


@njit
def bitsum(a):
    s = 0.0
    for elem in a:
        s += bitsum_lookup[elem]
    return s


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

