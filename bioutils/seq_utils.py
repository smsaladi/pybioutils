"""
Some utility functions for dealing with sequences

# do it this way since we want to maintain the order of the alignment
keep_rows = aln.index.map(lambda x: x in keep_to_show)
"""

import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment


def filter_non_gapped(aln):
    """provide alignment as values of a pd.Series
    """

    aln_arr = np.array(aln.apply(list).tolist())

    def has_char(x):
        return np.any(x != '-') & np.any(x != '.')
    keep_positions = np.apply_along_axis(has_char, 0, aln_arr)

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

