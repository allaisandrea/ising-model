#!/usr/bin/env python3
import sys
import pandas
import json
import subprocess
import os


def compute_aggregate_observables(json_db, dest_file_name):
    file_names_table = pandas.DataFrame(**json_db["file_names"])
    reweight_args_table = pandas.DataFrame(**json_db["reweight_args"])
    try:
        os.remove(dest_file_name)
    except OSError:
        pass

    for row in reweight_args_table.itertuples():
        file_names = file_names_table.loc[
            file_names_table['group_id'] == row.group_id, 'file_name']
        if len(file_names) == 0:
            print("No entry in \"file_names\" "
                  "table for group {}\n".format(row.group_id))
            continue

        command = [
            'compute-aggregate-observables',
            '--mu', str(row.mu),
            '--J-begin', str(round(row.J - row.dJ, 6)),
            '--J-end', str(round(row.J + row.dJ, 6)),
            '--n-J', '128',
            '--measure-every', str(row.measure_every),
            '--skip-first-n', str(row.skip_first_n),
            '--out-file', dest_file_name,
            '--file-group', str(row.group_id),
            '--files'] + list(file_names)
        print(' '.join(command))
        subprocess.call(command)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage:\ncompute-aggregate-observables.py "
                 "file_groups.json autocorrelation.bin")

    with open(sys.argv[1]) as in_file:
        json_db = json.load(in_file)

    compute_aggregate_observables(json_db, sys.argv[2])
