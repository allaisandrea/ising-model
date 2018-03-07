#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
from collections import namedtuple
import argparse
import pandas as pd
import numpy as np
import serialization
from utils import cross_validate, autocorrelation


def get_shape_field_names():
    return ["shape0", "shape1",  "shape2", "shape3", "shape4"]


def get_file_metadata_table(input_path):
    file_names = glob.glob(os.path.join(input_path, "*.bin"))
    metadata_fields = list(serialization.Metadata1._fields)
    metadata_fields.remove('wave_numbers')
    metadata_fields.remove('shape')
    shape_fields = get_shape_field_names()
    metadata_list = list()
    for file_name in file_names:
        with open(file_name) as in_file:
            metadata = serialization.read_metadata(in_file)
        statinfo = os.stat(file_name)
        base_name = os.path.basename(file_name)
        shape = [1] * 5
        for i, shape_i in enumerate(metadata.shape):
            shape[i] = shape_i
        metadata = [getattr(metadata, tag) for tag in metadata_fields]
        metadata_list.append([base_name, statinfo.st_size] + shape + metadata)

    column_names = ["file_name", "file_size"]
    column_names += shape_fields + metadata_fields
    table = pd.DataFrame(metadata_list, columns=column_names)

    return table.sort_values(['prob'] + shape_fields + ['file_size'])


def find_duplicates(file_metadata_table):
    unique_field_names = get_shape_field_names() + ["prob", "seed"]
    def get_unique_fields(item):
        return [getattr(item, name) for name in unique_field_names]

    sorted_table = file_metadata_table.sort_values(unique_field_names)
    duplicates = list()
    previous_item = None
    for item in sorted_table.itertuples():
        unique_fields = get_unique_fields(item)
        if previous_item is not None:
            previous_unique_fields = get_unique_fields(previous_item)
            if unique_fields == previous_unique_fields:
                duplicates.append(previous_item)
                duplicates.append(item)
        previous_item = item
    return pd.DataFrame(duplicates)


AccumulatedObservables = namedtuple(
    'AccumulatedObservables',
    ['wave_numbers',
     'magnetization',
     'fourier_transform_2d'])


def make_new_accumulated_observables():
    return AccumulatedObservables(
        wave_numbers=list(),
        magnetization=list(),
        fourier_transform_2d=list())


def append_observables(path,
                       measure_every,
                       file_metadata,
                       accumulated_observables):
    if measure_every % file_metadata.measure_every != 0:
        print("Non multiple 'measure_every' for file {}".format(
            file_metadata.file_name))
        return
    step = measure_every // file_metadata.measure_every
    skip = 32

    with open(os.path.join(path, file_metadata.file_name)) as in_file:
        metadata, observables = serialization.read_file(in_file)

    if len(observables.magnetization) < skip * step:
        print("Skipping file {}: too few observables".format(
            file_metadata.file_name))
        return
    accumulated_observables.magnetization.extend(
        observables.magnetization[skip * step:-1:step])
    accumulated_observables.fourier_transform_2d.extend(
        observables.fourier_transform_2d[skip * step:-1:step])
    if len(accumulated_observables.wave_numbers) == 0:
        accumulated_observables.wave_numbers.extend(metadata.wave_numbers)
    elif accumulated_observables.wave_numbers != list(metadata.wave_numbers):
        print("Wave numbers do not match for file {}: {} {}".format(
            file_metadata.file_name,
            accumulated_observables.wave_numbers,
            metadata.wave_numbers))

def append_summary(unique_fields, observables, scalars_list, momenta_list):
    def compute_Phi2_inv(phi2):
        return 1.0 / np.mean(phi2)
    def compute_Phi2_2(phi2):
        return np.square(np.mean(phi2))
    def compute_Phi4(phi2):
        return np.mean(np.square(phi2))
    def compute_lambdaR(phi2):
        Phi2 = np.mean(phi2)
        Phi2_2 = np.square(Phi2)
        Phi4 = np.mean(np.square(phi2))
        return  (3.0 * Phi2_2 - Phi4) / np.power(Phi2, 4.0)

    shape = unique_fields[1:6]
    volume = 1
    for L in shape:
        volume *= L

    m = np.array(observables.magnetization)
    m2 = np.square(m)
    m2_ac = cross_validate(lambda x: autocorrelation(x, 1), m2, 16)
    Phi2_arr = volume * m2;
    Phi2_inv = cross_validate(compute_Phi2_inv, Phi2_arr, 16)
    Phi2_2 = cross_validate(compute_Phi2_2, Phi2_arr, 16)
    Phi4 = cross_validate(compute_Phi4, Phi2_arr, 16)
    lambdaR = volume * cross_validate(compute_lambdaR, Phi2_arr, 16)
    scalars = list(unique_fields)
    scalars.append(m2_ac[0])
    scalars.append(np.abs(m2_ac[0]) / m2_ac[1])
    scalars.extend(Phi2_inv)
    scalars.extend(Phi2_2)
    scalars.extend(Phi4)
    scalars.extend(lambdaR)
    scalars_list.append(scalars)

    def compute_Phi2k_inv(ftsq):
        ftsq_mean = np.mean(ftsq, axis=0)
        return 1.0 / np.concatenate([
            ftsq_mean[0:1, 0], 0.5 * (ftsq_mean[1::2, 0] + ftsq_mean[2::2, 0])])

    Phi2k_arr = volume * np.square(np.stack(observables.fourier_transform_2d))
    Phi2k_inv = cross_validate(compute_Phi2k_inv, Phi2k_arr, 16)

    wave_numbers = [0] + observables.wave_numbers
    for i in range(Phi2k_inv.shape[0]):
        momenta_row = list(unique_fields)
        momenta_row.append(wave_numbers[i])
        momenta_row.append(2 * np.pi * wave_numbers[i] / shape[0])
        momenta_row.extend(Phi2k_inv[i])
        momenta_list.append(momenta_row)

def get_momenta_column_names():
    return [
        "wave_number",
        "momentum",
        "Phi2k_inv",
        "Phi2k_inv_s",
        ]

def get_scalars_column_names():
    return [
        "m2_ac",
        "m2_ac_s",
        "Phi2_inv",
        "Phi2_inv_s",
        "Phi2_2",
        "Phi2_2_s",
        "Phi4",
        "Phi4_s",
        "lambdaR",
        "lambdaR_s"
    ]


def main(path):
    unique_field_names = ['prob'] + get_shape_field_names()

    file_metadata_table = get_file_metadata_table(path)
    duplicates = find_duplicates(file_metadata_table)
    if len(duplicates) > 0:
        sys.stderr.write("Found duplicate files\n")
        duplicates.to_csv(sys.stderr)
        return
    else:
        file_metadata_table.to_csv(os.path.join(path, "file_metadata.csv"))


    file_metadata_table = file_metadata_table.sort_values(
        unique_field_names + ['measure_every'],
        ascending=([True] * len(unique_field_names) + [False]))

    #file_metadata_table = file_metadata_table[
        #file_metadata_table['n_measure'] < 8*1024]

    previous_unique_fields = None
    measure_every = None
    accumulated_observables = make_new_accumulated_observables()
    scalars_list = list()
    momenta_list = list()
    for file_metadata in file_metadata_table.itertuples():
        unique_fields = [getattr(file_metadata, field_name)
                         for field_name in unique_field_names]
        if unique_fields != previous_unique_fields:
            if previous_unique_fields is not None:
                append_summary(previous_unique_fields,
                               accumulated_observables,
                               scalars_list,
                               momenta_list)
                accumulated_observables = make_new_accumulated_observables()
            measure_every = file_metadata.measure_every
        append_observables(path,
                           measure_every,
                           file_metadata,
                           accumulated_observables)
        previous_unique_fields = unique_fields

    append_summary(previous_unique_fields,
                   accumulated_observables,
                   scalars_list,
                   momenta_list)

    scalars_table = pd.DataFrame(
        scalars_list,
        columns=unique_field_names + get_scalars_column_names())
    momenta_table = pd.DataFrame(
        momenta_list,
        columns=unique_field_names + get_momenta_column_names())
    scalars_table.to_csv(os.path.join(path, "scalars.csv"))
    momenta_table.to_csv(os.path.join(path, "momenta.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compile table of metadata.')
    parser.add_argument('path', type=str, help='Path to the data')
    args = parser.parse_args()
    main(**vars(args))
