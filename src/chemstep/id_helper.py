import numpy as np
from numba import njit


tranche_dict = {
    "0": [0, "M500"], "1": [1, "M400"], "2": [2, "M300"], "3": [3, "M200"], "4": [4, "M100"], "5": [5, "M000"],
    "6": [6, "P000"], "7": [7, "P010"], "8": [8, "P020"], "9": [9, "P030"], "a": [10, "P040"], "b": [11, "P050"],
    "c": [12, "P060"], "d": [13, "P070"], "e": [14, "P080"], "f": [15, "P090"], "g": [16, "P100"], "h": [17, "P110"],
    "i": [18, "P120"], "j": [19, "P130"], "k": [20, "P140"], "l": [21, "P150"], "m": [22, "P160"], "n": [23, "P170"],
    "o": [24, "P180"], "p": [25, "P190"], "q": [26, "P200"], "r": [27, "P210"], "s": [28, "P220"], "t": [29, "P230"],
    "u": [30, "P240"], "v": [31, "P250"], "w": [32, "P260"], "x": [33, "P270"], "y": [34, "P280"], "z": [35, "P290"],
    "A": [36, "P300"], "B": [37, "P310"], "C": [38, "P320"], "D": [39, "P330"], "E": [40, "P340"], "F": [41, "P350"],
    "G": [42, "P360"], "H": [43, "P370"], "I": [44, "P380"], "J": [45, "P390"], "K": [46, "P400"], "L": [47, "P410"],
    "M": [48, "P420"], "N": [49, "P430"], "O": [50, "P440"], "P": [51, "P450"], "Q": [52, "P460"], "R": [53, "P470"],
    "S": [54, "P480"], "T": [55, "P490"], "U": [56, "P500"], "V": [57, "P600"], "W": [58, "P700"], "X": [59, "P800"],
    "Y": [60, "P900"], "Z": [61, None]
}


z22_lookup = np.zeros((123, 10), dtype=np.int64)

for char in tranche_dict:
    for idx in range(10):
        v = tranche_dict[char][0]
        z22_lookup[ord(char), idx] = v * 62**idx


z22_revlookup = dict()

for char in tranche_dict:
    v = tranche_dict[char][0]
    for idx in range(10):
        z22_revlookup[v * 62**idx] = char


@njit
def convert_chars(s):
    chars = np.zeros(len(s), dtype=np.int32)
    for i, charac in enumerate(s):
        chars[i] = ord(charac)
    return chars


def char_to_int64(s, prefix="CSLB"):
    s = s.strip()
    if len(s) == 10 + len(prefix):
        s = s[len(prefix):]
    if len(s) != 10:
        raise ValueError(f"This is not a standard {prefix} ID: {s}")
    else:
        return z2int_helper(s)


def int64_to_char(zint, prefix="CSLB"):
    s = ""
    for i in range(10):
        val = zint % 62
        s = z22_revlookup[val] + s
        zint = (zint - val) // 62
    s = prefix + s
    return s


@njit
def z2int_helper(s):
    val = 0
    for i in range(9, -1, -1):
        val += z22_lookup[ord(s[i]), 9-i]
    return val
