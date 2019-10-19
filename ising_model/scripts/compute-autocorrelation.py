#!/usr/bin/env python3
import sys
import pandas

def main(argv):
    if len(argv) != 2:
        sys.stderr.write("Usage compute-autocorrelation.py file_groups.yaml")
        return -1;

    file_groups = pandas.read_csv(argv[1], skipinitialspace=True)
    for key, group in file_groups.groupby('group_id'):
        print(group)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
