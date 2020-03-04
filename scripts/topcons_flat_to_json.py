"""
This script converts the "flat" text output from [TOPCONS](http://topcons.net/)
("Dumped prediction in one text file")
into a newline-delimited json file

Author: Shyam Saladi
Date: March 2020
License: MIT

"""

import sys
import itertools
import re
from collections import OrderedDict

import argparse

import json
from json import JSONEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_txt')

    args = parser.parse_args()

    if args.result_txt == '-':
        fh = sys.stdin
    else:
        fh = open(args.result_txt, 'r')

    for entry in read_topcons(fh):
        print(json.dumps(entry))
    
    if fh is not sys.stdin:
        fh.close()

    return

def read_topcons(fh):
    """yeilds a single entry at a time
    Each entry is separated by 78 pound signs
    """
    file_it = read_chunks(fh, sep='#' * 78)
    # skip informational part at the beginning of the file
    next(file_it, None)
    next(file_it, None)

    for chunk in file_it:
        chunk = re.sub('\n+', '\n', chunk).replace(':\n', ': ')

        summary, *data_sections = chunk.split('\nPredicted ')

        # Empty section (probably end of file)
        if summary == '\n':
            continue

        data = OrderedDict()
        for line in summary.lstrip().split('\n'):
            k, v = line.split(': ', maxsplit=1)
            data[k] = v

        for section in data_sections:
            name, *values = section.split('\n')
            data['Predicted ' + name] = [v.split() for v in values]

        yield data

    return

def read_chunks(fh, sep, chunk_size=4096):
    """Reads a file with a custom separator
    https://stackoverflow.com/a/16260159/2320823
    """
    buf = ""

    while True:
        while sep in buf:
            pos = buf.index(sep)
            yield buf[:pos]

            buf = buf[pos + len(sep):]

        chunk = fh.read(chunk_size)
        if not chunk:
            yield buf
            break

        buf += chunk

    return


if __name__ == '__main__':
    main()

