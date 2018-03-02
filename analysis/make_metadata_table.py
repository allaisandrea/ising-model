#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import os
import glob
import serialization
import argparse
import pandas as pd


def main(input_path, output):
    metadata_table = serialization.get_metadata_table(input_path)
    metadata_table.to_csv(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compile table of metadata.')
    parser.add_argument('input_path', type=str, help='Path to the data')
    parser.add_argument('output', type=str, help='Output path')
    args = parser.parse_args()
    main(**vars(args))
