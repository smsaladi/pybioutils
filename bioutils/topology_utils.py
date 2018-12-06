"""Some utilities to help deal with and process

Author: Shyam Saladi (saladi@caltech.edu)
August 2018

All code contained herein is licensed under an
[MIT License](https://opensource.org/licenses/MIT)
"""


import re
import itertools

import numba
import numpy as np
import pandas as pd

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

def test_last_tmd():
    topo = 'iiiiiiiiMMMMMMMMMMMMMMMMMMMMMooooooooooo'
    assert np.array_equal(_last_tmd(topo), [8, 29])
    topo = 'MMMMMMMMMMMMMMMMMMMMMooooooooooo'
    assert np.array_equal(_last_tmd(topo), [0, 21])
    topo = 'MMMMMMMMMMMMMMMMMMMMMiiiiiiiiiii'
    assert np.array_equal(_last_tmd(topo), [0, 21])
    topo = 'iiiiiiiiiiiMMMMMMMMMMMMMMMMMMMMM'
    assert np.array_equal(_last_tmd(topo), [11, 32])
    topo = 'oooooooooooMMMMMMMMMMMMMMMMMMMMM'
    assert np.array_equal(_last_tmd(topo), [11, 32])

def _last_tmd(topo):
    """Returns the start and stop indicies of the last TMD"""
    last_topo = topo.rstrip('-')
    # Some issue if this is case...
    if len(last_topo) == 0:
        return -1, -1
    else:
        last_topo = last_topo[-1]

    tm_stop = topo.rfind('M') + 1

    # no TM present
    if tm_stop == -1:
        return -1, -1

    if last_topo == 'M':
        if 'i' in topo and 'o' in topo:
            tm_start = min(topo.rfind('i'), topo.rfind('o')) + 1
        elif 'i' in topo:
            tm_start = topo.rfind('i') + 1
        elif 'o' in topo:
            tm_start = topo.rfind('o') + 1
        else:
            raise ValueError("Neither i nor o in topology")
    elif last_topo in ['i', 'o']:
        if last_topo == 'i':
            tm_start = topo.rfind('o') + 1
        elif last_topo == 'o':
            tm_start = topo.rfind('i') + 1
    else:
        print(row.to_dict(), file=sys.stderr)
        raise ValueError("Last topology unrecognized: " + last_topo +
                         "\nin above row")

    return tm_start, tm_stop

test_last_tmd()

@numba.jit()
def last_tmd(row, topo_col='topostring', seq_col='seq'):
    """Returns the sequence of the last TMD"""
    topo = row[topo_col]
    seq = row[seq_col]
    
    tm_start, tm_stop = _last_tmd(topo)
    return seq[tm_start:tm_stop]

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
