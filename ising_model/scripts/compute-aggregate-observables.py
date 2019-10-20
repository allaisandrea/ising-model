#!/usr/bin/env python3
import sys
import pandas
import json
import subprocess
import os


def compute_aggregate_observables(json_db, dest_file_name):
    file_names_table = pandas.DataFrame(**json_db["file_names"])
    measure_every_table = pandas.DataFrame(**json_db["measure_every"])
    reweight_range_table = pandas.DataFrame(**json_db["reweight_range"])
    try:
        os.remove(dest_file_name)
    except OSError:
        pass

    for row in reweight_range_table.itertuples():
        file_names = file_names_table.loc[
            file_names_table['group_id'] == row.group_id, 'file_name']
        if len(file_names) == 0:
            sys.error("No entry in \"file_names\" "
                      "table for group {}\n".format(row.group_id))
            continue

        measure_every = measure_every_table.loc[
            measure_every_table['group_id'] == row.group_id, 'measure_every']
        if len(measure_every) != 1:
            sys.error("No entry or multiple entries in \"measure_every\" "
                      "table for group {}\n".format(row.group_id))
            continue
        command = [
            'compute-aggregate-observables',
            '--mu', str(row.mu),
            '--J-begin', str(row.J_begin),
            '--J-end', str(row.J_end),
            '--n-J', str(row.n_J),
            '--measure-every', str(measure_every[0]),
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
