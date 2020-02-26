"""
A set of functions for filtering and munging alignments

see `parse_files.read_seqio` monkeypatched to `pd.Series.from_seqio`

# do it this way since we want to maintain the order of the alignment
keep_rows = aln.index.map(lambda x: x in keep_to_show)
"""

import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

def has_char(x):
    return np.any(x != '-') & np.any(x != '.')

def measure_gap_pct(x):
    return np.sum((x == '-') | (x == '.')) / x.size
    
def keep_ref_positions(aln, refids, highest_gap_pct=0.5):
    """provide alignment as values of a pd.Series
    """
    
    aln_arr = np.array(aln.apply(list).tolist())
    refids = aln.index[aln.index.isin(refids)]
    aln_arr_ref = np.array(aln.loc[refids].apply(list).tolist())

    keep_positions = np.apply_along_axis(has_char, 0, aln_arr_ref)
    gap_pct = np.apply_along_axis(measure_gap_pct, 0, aln_arr)
    
    good_pos = gap_pct < highest_gap_pct
    aln_arr = aln_arr[:, keep_positions | good_pos]

    # join back into a single string
    aln_arr =  np.apply_along_axis(lambda x: "".join(x), 1, aln_arr)

    return pd.Series(index = aln.index,
                     name = aln.name,
                     data = aln_arr)


# filter based on reference
def find_ref_positions(aln, refids, highest_gap_pct):
    aln_arr = np.array(aln.apply(list).tolist())
    refids = aln.index[aln.index.isin(refids)]
    aln_arr_ref = np.array(aln.loc[refids].apply(list).tolist())

    keep_positions = np.apply_along_axis(has_char, 0, aln_arr_ref)
    gap_pct = np.apply_along_axis(measure_gap_pct, 0, aln_arr)
    
    good_pos = gap_pct < highest_gap_pct
    return keep_positions | good_pos


def extract_positions(aln, keep_positions):
    aln_arr = np.array(aln.apply(list).tolist())
    aln_arr = aln_arr[:, keep_positions]

    # join back into a single string
    aln_arr =  np.apply_along_axis(lambda x: "".join(x), 1, aln_arr)

    return pd.Series(index = aln.index,
                     name = aln.name,
                     data = aln_arr)


def format_aln(aln, format):
    """Takes an alignment as a pd.Series
       and returns a string in clustal format
    """

    bp_aln = MultipleSeqAlignment([
            SeqRecord(Seq(v), id=k) for k, v in aln.items()
        ])

    return bp_aln.format(format)

