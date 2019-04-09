"""Utilities to parse/write various files

    * metadata table from Uniprot
    * fasta files
    * JSON file of TOPCONS predictions
    * SignalP output file

Author: Shyam Saladi (saladi@caltech.edu)
August 2018

All code contained herein is licensed under an
[MIT License](https://opensource.org/licenses/MIT)
"""

import os.path
import re
import gzip
import json
import functools

import numpy as np
import pandas as pd

from Bio import SeqIO
import Bio.PDB

def to_fasta(self, fn=None):
    """Monkey patch pd.Series to provide a to_fasta method"""
    if fn:
        with open(fn, 'w') as fh:
            for idx, val in self.iteritems():
                print('>{}\n{}'.format(idx, val), file=fh)
    else:
        str = ''
        for idx, val in self.iteritems():
            str += '>{}\n{}\n'.format(idx, val)
        return str
pd.Series.to_fasta = to_fasta


@classmethod
def series_seqio(cls, fn, *args, **kwargs):
    if isinstance(fn, str) and 'gz' in fn:
        with gzip.open(fn, "rt") as fh:
            seqs = [(r.id, str(r.seq)) for r in SeqIO.parse(fh, *args, **kwargs)]
    else:
        seqs = [(r.id, str(r.seq)) for r in SeqIO.parse(fn, *args, **kwargs)]
    seqs = list(zip(*seqs))
    seqs = cls(seqs[1], index=seqs[0], name="seq")
    return seqs
pd.Series.from_seqio = series_seqio

"""

Some utilities to deal Uniprot sequence and metadata

"""


def read_uniprot(excel_fn, faa_fn):
    """Reads a uniprot table as well as an associated fasta file"""

    df = pd.read_excel(excel_fn)

    # if the field is empty, it's read in as np.nan
    # just replace with empty string
    for col in ['Fragment', 'Transmembrane', 'Signal peptide', 'Transit peptide',
                'Compositional bias', 'Caution', 'Sequence caution',
                'Subcellular location [CC]', 'Gene ontology (cellular component)',
                'Cross-reference (Pfam)']:
        df.loc[df[col].isnull(), col] = ''

    # read (accomidating .gz compressed file) and join amino acid sequences
    if 'gz' in faa_fn:
        with gzip.open(faa_fn, "rt") as fh:
            seqs = {r.id:str(r.seq) for r in SeqIO.parse(fh, "fasta")}
    else:
        seqs = {r.id:str(r.seq) for r in SeqIO.parse(faa_fn, "fasta")}

    seqs = pd.Series(seqs, name="seq")
    seqs.index = seqs.index.str.split('|').str[1]

    df = df.merge(seqs.to_frame(), left_on="Entry", right_index=True)

    return df


"""

Deal with hmmsearch

"""

def read_hmmsearch_dom(fn):
    """Read the output of hmmsearch
    
    Since the description field can have anything in it, if it has spaces, pandas croaks.
    As a result, we have to read the file in a bit of a roundabout way
    """
    cols = ['target name', 'target accession', 'tlen',
            'query name', 'query accession', 'qlen',
            'full_E-value', 'full_score', 'full_bias',
            'domain_idx', 'domain_hits',
            'domain_c-Evalue', 'domain_i-Evalue', 'domain_score', 'domain_bias',
            'hmm_from', 'hmm_to', 'ali_from', 'ali_to', 'env_from', 'env_to',
            'accuracy', 'description of target'
        ]
    df = pd.read_csv(fn, sep="^", header=None, comment='#')
    df = df[0].str.split('\s+', n=len(cols)-1, expand=True)
    df.columns = cols
    
    for c in ['full_E-value', 'full_score', 'full_bias',
              'domain_c-Evalue', 'domain_i-Evalue',
              'domain_score', 'domain_bias', 'accuracy']:
        df[c] = df[c].astype(float)
        
    for c in ['tlen', 'qlen',
              'domain_idx', 'domain_hits',
              'hmm_from', 'hmm_to',
              'ali_from', 'ali_to',
              'env_from', 'env_to']:
        df[c] = df[c].astype(int)

    # convert to 0-based indexing
    for c in ['hmm_from', 'ali_from', 'env_from']:
        df[c] -=1

    return df

"""

Some utilities to deal with predictions from TOPCONS

"""

def read_topcons(fn):
    """Reads topcons predictions into a pd.DataFrame.

    Needs the predictions in a json format (not TOPCONS's raw text format)
    Use `parse_topcons.py` to carry out this conversion

    Cleans up sequence names (expecting something like `sp|Q3BBV1|NBPFK_HUMAN`) into
    just `Q3BBV1`

    """
    if 'json' not in fn:
        raise ValueError("Convert to json first")

    if fn.endswith('gz'):
        op = functools.partial(gzip.open, encoding='UTF-8')
    else:
        op = open

    with op(fn, 'rt') as fh:
        data = json.load(fh)

    df = pd.concat([parse_topopred(x) for x in data], axis=1, sort=False)
    df = df.T

    # Clean up names
    df[['Entry', 'desc']] = df['Sequence name'].str.split('\s', n=1, expand=True)
    df[['sptr', 'Entry', 'name']] = df['Entry'].str.split('\|', n=2, expand=True)

    df.drop(columns=['Sequence name', 'Sequence number',
                     'sptr', 'name', 'desc'], inplace=True)

    return df


def parse_topopred(pred, keep='TOPCONS predicted topology'):
    """Converts a single topology prediction to a pd.Series

        keep: str, list of str, optional
            Predictions to keep for each sequence,
            one or more of the following:
            ['TOPCONS predicted topology', 'OCTOPUS predicted topology',
             'Philius predicted topology', 'PolyPhobius predicted topology',
             'SCAMPI predicted topology', 'SPOCTOPUS predicted topology',
             'Homology']
    """

    if isinstance(keep, str):
        keep = [keep]

    seqlen = len(pred['TOPCONS predicted topology'])

    data = {}
    for k, v in pred.items():
        if k in ['Sequence number', 'Sequence name', 'Sequence']:
            data[k] = v.strip()
        elif k in keep:
            k = k.replace(' predicted topology', '')
            data[k] = clean_topo(v)
        # this is how we can identify predictions from homology
        elif ('Homology' in keep
              and not k.startswith('Predicted')
              and isinstance(v, str)
              and len(v) == seqlen):
            data['Homology'] = clean_topo(v)

    return pd.Series(data)


def clean_topo(topo):
    """Cleans topology string by removing odd/unnecessary characters

    Also validates topostring to make confirm that we know what we are working with
    """
    topo = re.sub('\s+', '', topo)

    if topo.startswith("***No"):
        return ""

    topo = topo.replace('L', '-').replace('u', '-')

    if not set(list(topo)).issubset(['S', 'i', 'o', 'M', '-']):
        raise ValueError("Unexpected character in topostring: " + topo)

    return topo


def read_signalp(fn):
    """Reads the short output format from signalp
    """
    df = pd.read_table(fn, sep='\s+', skiprows=1)
    cols = df.columns.tolist()[1:]
    df.drop(columns=cols[-1], inplace=True)
    df.columns = cols
    df['hasSP'] = df['?'] == 'Y'
    df.drop(columns='?', inplace=True)
    return df



def read_cdhit(fn):
    data = []
    cur_cluster = None
    with open(fn, 'r') as fh:
        for line in fh.readlines():
            if line.startswith('>'):
                _, cur_cluster = line.rsplit(maxsplit=1)
            else:
                data.append([cur_cluster, line])
    df = pd.DataFrame(data, columns=['cluster', 'line'])

    df['cluster'] = df['cluster'].astype(int)

    df[['entry', 'ident']] = (df['line'].str.strip()
                            .str.split('>')
                            .str[1]
                            .str.replace('...', '', regex=False)
                            .str.replace(' at ', ' ', regex=False)
                            .str.split(expand=True))

    df['rep'] = df['ident'] == '*'

    if df['ident'].str.contains('/').any():
        df['strand'] = 1
        df.loc[df['ident'].str.contains('-/'), 'strand'] = -1
        df['ident'] = df['ident'].str.replace('[+-]/', '')

    df['ident'] = (df['ident'].str.replace('%', '', regex=False)
                              .str.replace('\s', '')
                              .str.replace('*', '100', regex=False)
                              .astype(float) / 100
                              )

    df.drop(columns='line', inplace=True)

    return df


def read_interpro(fn, seqfn=None):
    """Read files with a format that corresponds to protein2ipr.dat.gz
    Interpro's mapping of uniprot identifiers to Interpro domains
    See more: https://www.ebi.ac.uk/interpro/download.html
    """
    df = pd.read_table(fn,
                       header=None,
                       names=['uniprot', 'ipr', 'desc', 'sig', 'start', 'stop'])
    df['start'] -= 1
    
    if seqfn:
        df_seq = pd.Series.from_seqio(seqfn, 'fasta')
       
    df = df.join(df_seq, on="uniprot")
    return df


def read_ss2(fn):
    df = pd.read_table(fn, sep="\s+", comment='#',
                       names=['idx', 'aa', 'ss3', 'score_c', 'score_h', 'score_e'])
    df['fn'] = fn
    return df
    

def read_horiz(fn):
    with open(fn, "r") as fh:
        lines = [x.strip() for x in fh.readlines()]
        
    conf = [x.replace("Conf: ", "") for x in lines[2::6]]
    pred = [x.replace("Pred: ", "") for x in lines[3::6]]
    aa   = [x.replace("AA: ", "") for x in lines[4::6]]    
    
    conf = "".join(conf)
    pred = "".join(pred)
    aa   = "".join(aa)
    
    name = fn.split('/')[-1].split('.')[0]
    return pd.Series({'pred': pred, 'aa': aa, 'conf': conf}, name=name)


def read_dssp(fn):
    """reads output from dssp into a pandas dataframe"""
    data = Bio.PDB.make_dssp_dict(fn)
    df = pd.DataFrame.from_dict(data[0], orient='index')
    df.columns = ['aa', 'ss', 'asa', 'phi', 'psi', 'dssp_idx',
        'NH-->O_1_relidx', 'NH-->O_1_energy',
        'O-->NH_1_relidx', 'O-->NH_1_energy',
        'NH-->O_2_relidx', 'NH-->O_2_energy',
        'O-->NH_2_relidx', 'O-->NH_2_energy']

    df.drop(columns='dssp_idx', inplace=True)
    df['resid'] = df.index.str[1].str[1]
    df.reset_index(drop=True, inplace=True)

    return df

