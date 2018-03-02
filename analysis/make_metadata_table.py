#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import serialization
import argparse
import pandas as pd


def main(input_path):
    metadata_table = serialization.get_metadata_table(input_path)
    duplicates = serialization.find_duplicates(metadata_table)
    if len(duplicates) > 0:
        sys.stderr.write("Found duplicate files\n")
        duplicates.to_csv(sys.stderr)
    else:
        metadata_table.to_csv(sys.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compile table of metadata.')
    parser.add_argument('input_path', type=str, help='Path to the data')
    args = parser.parse_args()
    main(**vars(args))
