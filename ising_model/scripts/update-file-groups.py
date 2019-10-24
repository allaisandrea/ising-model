#!/usr/bin/env python3
import sys
import json
from ising_model import udh
import pandas
import glob
import os


def update_file_groups(directory, json_db):
    file_names = glob.glob(directory + "/*.udh")
    parameters_table = udh.load_params_table(file_names)
    file_names_table = pandas.DataFrame(**json_db["file_names"])
    existing_files = set(file_names_table['file_name'])
    if len(file_names_table) > 0:
        group_id = max(file_names_table['group_id']) + 1
    else:
        group_id = 0
    for row in parameters_table.itertuples():
        file_name = os.path.basename(row.file_name)
        if file_name in existing_files:
            continue
        json_db["file_names"]["data"].append(
            [group_id, file_name])
        json_db["measure_every"]["data"].append(
            [group_id, row.measure_every])
        json_db["reweight_range"]["data"].append(
            [group_id, row.mu, round(row.J - 1.0 / row.L0, 5),
             round(row.J + 1.0 / row.L0, 5), 128])
        group_id += 1
    return json_db


def print_data(data, str_format, stream):
    for i, row in enumerate(data):
        stream.write("      [" + str_format.format(*row) + "]")
        if i + 1 == len(data):
            stream.write("\n")
        else:
            stream.write(",\n")


def print_table(name, fields, stream):
    stream.write("  \"" + name + "\": {\n")
    stream.write("    \"columns\": " + json.dumps(fields["columns"]) + ",\n"),
    stream.write("    \"data\": [\n")
    if name == "file_names":
        print_data(fields["data"], "{:4}, \"{:>20}\"", stream)
    elif name == "measure_every":
        print_data(fields["data"], "{:4}, {:4}", stream)
    elif name == "reweight_range":
        print_data(fields["data"],
                   "{:4}, {:9.6f}, {:9.6f}, {:9.6f}, {:4}", stream)
    stream.write("    ]\n")
    stream.write("  }")


def print_json_db(json_db, stream):
    stream.write("{\n")
    for i, (key, value) in enumerate(json_db.items()):
        print_table(key, value, stream)
        if i + 1 == len(json_db):
            stream.write("\n")
        else:
            stream.write(",\n")
    stream.write("}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "Usage:\ncompute-autocorrelation.py directory file_groups.json")

    with open(sys.argv[2]) as in_file:
        json_db = json.loads(in_file.read())

    json_db = update_file_groups(sys.argv[1], json_db)
    print_json_db(json_db, sys.stdout)
