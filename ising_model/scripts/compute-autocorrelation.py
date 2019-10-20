#!/usr/bin/env python3
import sys
import pandas
import json
import subprocess
import os


def compute_autocorrelation(json_db, dest_file_name):
    file_names_table = pandas.DataFrame(**json_db["file_names"])
    measure_every_table = pandas.DataFrame(**json_db["measure_every"])
    try:
        os.remove(dest_file_name)
    except OSError:
        pass

    for row in measure_every_table.itertuples():
        file_names = file_names_table.loc[
            file_names_table['group_id'] == row.group_id, 'file_name']
        if len(file_names) == 0:
            sys.error(
                "No entry in \"file_names\" table for group {}\n".format(row.group_id))
            continue
        command = [
            'compute-autocorrelation',
            '--measure-every', str(row.measure_every),
            '--out-file', dest_file_name,
            '--file-group', str(row.group_id),
            '--files'] + list(file_names)
        print(' '.join(command))
        subprocess.call(command)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "Usage:\ncompute-autocorrelation.py file_groups.json autocorrelation.bin")

    with open(sys.argv[1]) as in_file:
        json_db = json.load(in_file)

    compute_autocorrelation(json_db, sys.argv[2])
