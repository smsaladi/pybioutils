"""Some utilities to help deal with and process

Author: Shyam Saladi (saladi@caltech.edu)
August 2018

All code contained herein is licensed under an
[MIT License](https://opensource.org/licenses/MIT)
"""

# import sys
import re
import itertools

try:
    import numba
except:
    print("numba not found. Nothing will be jitted")
    class numba:
        @staticmethod
        def jit(*args, **kwargs):
            def inner(func):
                return func
            return inner

import numpy as np
# import pandas as pd

# from tqdm.autonotebook import tqdm
from tqdm import tqdm
tqdm.pandas()

def test_parse_positionwise():
    pass

def parse_positionwise(topostring):
    """Generic topology string of same length as protein sequence

    e.g. 'oooooooooo oooooooMMM MMMMMMMMMM MMMMMMMMii iiiiiiiii\n\n'
    """
    # from https://stackoverflow.com/a/34444401
    groups = itertools.groupby(topostring)
    return [label for label, group in groups]

@numba.jit()
def tm_count(topo):
    topo_short = parse_positionwise(topo)
    topo_short = "".join(topo_short)
    return topo_short.count('M')

@numba.jit()
def c_loop_len(topo):
    """Counts the length of the cterminal loop"""
    return len(topo) - topo.rfind('M')

_re_space = re.compile('\s+')
_re_nonM = re.compile('[^m]')
def prep_topo(topo):
    topo = _re_space.sub('', topo)
    topo = topo.lower()
    topo = _re_nonM.sub('x', topo)
    return topo

def test_first_tmd():
    topo = 'iiiiiiiiMMMMMMMMMMMMMMMMMMMMMooooooooooo'
    assert np.array_equal(_first_tmd(topo), [8, 29])
    assert np.array_equal(_first_tmd(topo, 1), [7, 30])
    topo = 'iiiiiiiiMMMMMMMMMMMMMMMMMMMMMooooooooooMMMMM'
    assert np.array_equal(_first_tmd(topo), [8, 29])
    assert np.array_equal(_first_tmd(topo, 1), [7, 30])
    topo = 'MMMMMMMMMMMMMMMMMMMMMooooooooooo'
    assert np.array_equal(_first_tmd(topo), [0, 21])
    assert np.array_equal(_first_tmd(topo, 1), [0, 22])
    topo = 'MMMMMMMMMMMMMMMMMMMMMiiiiiiiiiii'
    assert np.array_equal(_first_tmd(topo), [0, 21])
    assert np.array_equal(_first_tmd(topo, 1), [0, 22])
    topo = 'iiiiiiiiiiiMMMMMMMMMMMMMMMMMMMMM'
    assert np.array_equal(_first_tmd(topo), [11, 32])
    assert np.array_equal(_first_tmd(topo, 1), [10, 32])
    topo = 'oooooooooooMMMMMMMMMMMMMMMMMMMMM'
    assert np.array_equal(_first_tmd(topo), [11, 32])
    assert np.array_equal(_first_tmd(topo, 1), [10, 32])
    topo = 'oooooooooooMMMMMMMMMMMMMMMMMMMMMiiMMMM'
    assert np.array_equal(_first_tmd(topo), [11, 32])
    assert np.array_equal(_first_tmd(topo, 3), [8, 34])
    topo = 'MMMMooMMMMMMMMMMMMMMMMMMMMMii'
    assert np.array_equal(_first_tmd(topo), [0, 4])
    assert np.array_equal(_first_tmd(topo, 3), [0, 6])

def _first_tmd(topo, ext=0):
    topo = prep_topo(topo)

    tmd_start = topo.find('m')
    if tmd_start == -1:
        return -1, -1

    tmd_stop = topo.find('x', tmd_start)
    if tmd_stop == -1:
        tmd_stop = len(topo)

    if ext > 0:
        # make sure not to go into next TM
        next_m = topo.find('m', tmd_stop, tmd_stop + ext)
        if next_m == -1:
            # don't go past extents of sequence
            tmd_stop = min(tmd_stop + ext, len(topo))
        else:
            tmd_stop = next_m

        pre_m = topo.rfind('m', tmd_start - ext, tmd_start)
        if pre_m == -1:
            tmd_start = max(0, tmd_start - ext)
        else:
            tmd_start = pre_m+1
    return tmd_start, tmd_stop

@numba.jit()
def first_tmd(row, topo_col='topostring', seq_col='seq', ext=0):
    """Returns the sequence of the first TMD"""
    topo = row[topo_col]
    seq = row[seq_col]

    tm_start, tm_stop = _first_tmd(topo, ext) # **kwargs)
    return seq[tm_start:tm_stop]

test_first_tmd()

@numba.jit()
def nloop_len(topo):
    start = _first_tmd(topo)[0]
    if start == -1:
        return -1
    return start

def test_last_tmd():
    topo = 'iiiiiiiiMMMMMMMMMMMMMMMMMMMMMooooooooooo'
    assert np.array_equal(_last_tmd(topo), [8, 29])
    assert np.array_equal(_last_tmd(topo, 1), [7, 30])
    topo = 'MMMMMMMMMMMMMMMMMMMMMooooooooooo'
    assert np.array_equal(_last_tmd(topo), [0, 21])
    assert np.array_equal(_last_tmd(topo, 1), [0, 22])
    topo = 'MMMMMMMMMMMMMMMMMMMMMiiiiiiiiiii'
    assert np.array_equal(_last_tmd(topo), [0, 21])
    assert np.array_equal(_last_tmd(topo, 1), [0, 22])
    topo = 'iiiiiiiiiiiMMMMMMMMMMMMMMMMMMMMM'
    assert np.array_equal(_last_tmd(topo), [11, 32])
    assert np.array_equal(_last_tmd(topo, 1), [10, 32])
    topo = 'oooooooooooMMMMMMMMMMMMMMMMMMMMM'
    assert np.array_equal(_last_tmd(topo), [11, 32])
    assert np.array_equal(_last_tmd(topo, 1), [10, 32])
    topo = 'oooooooooooMMMMMMMMMMMMMMMMMMMMMiiMMMM'
    assert np.array_equal(_last_tmd(topo), [34, 38])
    assert np.array_equal(_last_tmd(topo, 3), [32, 38])
    topo = 'MMMMoooooooooooMMMMMMMMMMMMMMMMMMMMMii'
    assert np.array_equal(_last_tmd(topo), [15, 36])
    assert np.array_equal(_last_tmd(topo, 3), [12, 38])
    topo = 'MMMMooMMMMMMMMMMMMMMMMMMMMMii'
    assert np.array_equal(_last_tmd(topo), [6, 27])
    assert np.array_equal(_last_tmd(topo, 3), [4, 29])

def _last_tmd(topo, ext=0):
    """Returns the start and stop indicies of the last TMD"""
    topo = prep_topo(topo)

    tmd_stop = topo.rfind('m') + 1
    if tmd_stop == 0:
        return -1, -1

    tmd_start = topo.rfind('x', 0, tmd_stop) + 1
    if tmd_start == 0:
        tmd_start = 0

    if ext > 0:
        # make sure not to go into next TM
        next_m = topo.find('m', tmd_stop, tmd_stop + ext)
        if next_m == -1:
            # don't go past extents of sequence
            tmd_stop = min(tmd_stop + ext, len(topo))
        else:
            tmd_stop = next_m

        pre_m = topo.rfind('m', tmd_start - ext, tmd_start)
        if pre_m == -1:
            tmd_start = max(0, tmd_start - ext)
        else:
            tmd_start = pre_m+1

    return tmd_start, tmd_stop

test_last_tmd()

# @numba.jit()
def last_tmd(row, topo_col='topostring', seq_col='seq', ext=0): # **kwargs):
    """Returns the sequence of the last TMD"""
    topo = row[topo_col]
    seq = row[seq_col]

    tm_start, tm_stop = _last_tmd(topo, ext) # **kwargs)
    return seq[tm_start:tm_stop]

@numba.jit()
def cloop_len(topo):
    return len(topo) - _last_tmd(topo)[1]

@numba.jit()
def cterm_seq(row, topo_col='topostring', seq_col='seq'):
    """Returns the sequence of the last TMD"""
    topo = row[topo_col]
    seq = row[seq_col]

    tm_start, tm_stop = _last_tmd(topo)
    return seq[tm_stop:]


def test_has_comp_bias_tmd():
    pass

@numba.jit
def _has_comp_bias_tmd(bias, topo):
    tm_start, tm_stop = _last_tmd(topo)
    for seg, start, stop in re.findall(r'([A-Z]+)\s+(\d+)\s+(\d+).*?', bias):
        # use 0-/python indexing
        start = int(start) - 1
        stop = int(stop)
        # starts in region and ends anywhere
        if (start < tm_start <= stop or
            # starts before region but stops in region
            start < tm_stop <= stop or
            # starts before and ends after
            (tm_start < start and tm_stop >= stop)):
                return True
    return False

test_has_comp_bias_tmd()

def has_comp_bias_tmd(row, bias_col='Compositional bias', topo_col='topostring'):
    """Detects if the compositional bias overlaps with the last tmd"""
    bias = row[bias_col]
    topo = row[topo_col]
    return _has_comp_bias_tmd(bias, topo)


def test_uniprot_topo_to_long():
    out = _uniprot_topo_to_long("TRANSMEM 10 15 Helical. {ECO:0000255}.",
                                "HAHAHAHJDSFKASDFDSANFJLDASFN")
    assert out == "iiiiiiiiiMMMMMMooooooooooooo"

def _uniprot_topo_to_long(topo_data, seq):
    topo = np.array(list(seq))

    prev_loop = 'i'
    prev_stop = 0
    for seg, start, stop in re.findall(r'([A-Z]+)\s+(\d+)\s+(\d+).*?', topo_data):
        start = int(start)
        stop = int(stop)

        topo[prev_stop:start-1] = prev_loop
        if prev_loop == 'o':
            prev_loop = 'i'
        else:
            prev_loop = 'o'

        topo[start-1:stop] = 'M'
        prev_stop = stop

    topo[prev_stop:] = prev_loop
    return "".join(topo)

test_uniprot_topo_to_long()

def uniprot_topo_to_long(row, topo_col='Transmembrane', seq_col='seq'):
    """Converts uniprot's topology annotation intoa topology string to use the methods above
    Since loop information doesn't exist, just assume n-term inside
    """
    return _uniprot_topo_to_long(row[topo_col], row[seq_col])
