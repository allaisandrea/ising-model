#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import os
import glob
import yaml
import argparse
import pandas as pd
import numpy as np
import serialization
import utils


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
        with open(file_name, 'rb') as in_file:
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
    return pd.DataFrame(metadata_list, columns=column_names)


def find_duplicates(file_metadata_table):
    unique_fields = get_shape_field_names() + ["i_prob", "seed"]
    groups = file_metadata_table.groupby(unique_fields)
    duplicates = pd.DataFrame()
    for label, group in groups:
        if len(group) > 1:
            duplicates = duplicates.append(group)
    return duplicates


def extrapolate_prob(prob, ref_prob, parallel_count, magnetization, n_batches):
    def extract_quantiles(x):
        i0 = 0
        i1 = x.shape[-1] // 2
        i2 = -1
        return np.stack([x[:, i0], x[:, i1], x[:, i2]])

    assert len(parallel_count) == len(magnetization)

    n_prob = len(prob)
    n_obs = len(parallel_count)
    n_batch = n_obs // n_batches
    n_obs = n_batch * n_batches

    parallel_count = parallel_count[:n_obs]
    magnetization = magnetization[:n_obs]

    prob = np.reshape(prob, [n_prob, 1])
    max_parallel_count = np.amax(parallel_count)
    min_parallel_count = np.amin(parallel_count)
    parallel_count = np.array(np.broadcast_to(parallel_count, [n_prob, n_obs]))
    shift = np.where(
        np.greater(prob, ref_prob), min_parallel_count, max_parallel_count)
    parallel_count -= shift
    weights = np.power(prob / ref_prob, -parallel_count)

    weights = np.reshape(weights, [n_prob, n_batches, n_batch])
    weights_sum = np.sum(weights, axis=-1)

    magnetization = np.broadcast_to(magnetization, [n_prob, n_obs])
    magnetization = np.reshape(magnetization, [n_prob, n_batches, n_batch])

    m2_arr = np.square(magnetization)
    m2_arr_left = m2_arr[:, :, 1:]
    m2_arr_right = m2_arr[:, :, :-1]
    autocorr = np.sum(
        (m2_arr_left - np.mean(m2_arr_left, -1, keepdims=True)) *
        (m2_arr_right - np.mean(m2_arr_right, -1, keepdims=True)),
        axis=-1) / np.var(m2_arr, axis=-1) / (n_batch - 1)

    abs_m = np.sum(weights * np.abs(magnetization), axis=-1) / weights_sum
    m2 = np.sum(weights * m2_arr, axis=-1) / weights_sum
    m4 = np.sum(weights * np.square(m2_arr), axis=-1) / weights_sum
    chi = m2 - np.square(abs_m)
    U = 1.0 - m4 / (3.0 * np.square(m2))

    return {
        "ac": np.mean(autocorr, axis=-1),
        "ac_s": np.std(autocorr, axis=-1) / np.sqrt(n_batches),
        "m2": np.mean(m2, axis=-1),
        "m2_s": np.std(m2, axis=-1) / np.sqrt(n_batches),
        "chi": np.mean(chi, axis=-1),
        "chi_s": np.std(chi, axis=-1) / np.sqrt(n_batches),
        "U": np.mean(U, axis=-1),
        "U_s": np.std(U, axis=-1) / np.sqrt(n_batches)}


def append_observables_from_file(
        file_name, max_measure_every, n_skip, destination):

    with open(file_name, 'rb') as in_file:
        metadata, observables = serialization.read_file(in_file)

    if max_measure_every % metadata.measure_every != 0:
        print("Non multiple 'measure_every' for file {}".format(file_name))
        return

    step = max_measure_every // metadata.measure_every

    if len(observables.magnetization) < n_skip * step:
        print("Skipping file {}: too few observables".format(file_name))
        return

    field_names = list(destination.keys())
    field_names.pop(field_names.index('wave_numbers'))
    for field_name in field_names:
        destination[field_name].extend(
            getattr(observables, field_name)[n_skip * step:-1:step])

    if len(destination['wave_numbers']) == 0:
        destination['wave_numbers'].extend(metadata.wave_numbers)
    elif destination['wave_numbers'] != list(metadata.wave_numbers):
        print("Wave numbers do not match for file {}: {} {}".format(
            file_name,
            destination['wave_numbers'],
            metadata.wave_numbers))


def process_group(group, i_prob, shape0, ref_i_prob, path):
    observables = {
        'flip_cluster_duration': list(),
        'clear_flag_duration': list(),
        'measure_duration': list(),
        'wave_numbers': list(),
        'magnetization': list(),
        'parallel_count': list()}

    max_measure_every = group['measure_every'].max()

    for row in group.itertuples():
        assert row.shape1 == shape0 or row.shape1 == 1
        assert row.shape2 == shape0 or row.shape2 == 1
        assert row.shape3 == shape0 or row.shape3 == 1
        assert row.shape4 == shape0 or row.shape4 == 1

        append_observables_from_file(
            file_name=os.path.join(path, row.file_name),
            max_measure_every=max_measure_every,
            n_skip=32,
            destination=observables)

    aggregate_observables = extrapolate_prob(
        prob=i_prob,
        ref_prob=ref_i_prob,
        parallel_count=observables['parallel_count'],
        magnetization=observables['magnetization'],
        n_batches=16)

    column_dict = {
        'shape0': shape0,
        'i_prob': i_prob,
        'ref_i_prob': ref_i_prob,
        'measure_every': max_measure_every,
        'n_measure': len(observables['magnetization'])}

    for observable_name, observable in aggregate_observables.items():
        column_dict[observable_name] = observable

    return pd.DataFrame(column_dict)


def main(path, parameters_file_name, metadata_file_name):

    file_metadata_table = get_file_metadata_table(path)
    duplicates = find_duplicates(file_metadata_table)
    if len(duplicates) > 0:
        print("Found duplicate files")
        print(duplicates.to_csv())
        return
    else:
        file_metadata_table.to_csv(os.path.join(path, metadata_file_name))

    with open(os.path.join(path, parameters_file_name)) as in_file:
        parameters = yaml.safe_load(in_file)

    i_prob_ranges = list()
    for item in parameters['i_prob_ranges']:
        i_prob_ranges.append(
            [item[0]]+[utils.int_from_short_hex(i) for i in item[1:]])
    i_prob_ranges = [[item[0]] + list(map(utils.int_from_short_hex, item[1:]))
                     for item in parameters['i_prob_ranges']]

    scalars_table = pd.DataFrame()
    for shape0, ref_prob, min_prob, max_prob, step_prob in i_prob_ranges:
        group = file_metadata_table.loc[
            (file_metadata_table['i_prob'] == ref_prob) &
            (file_metadata_table['shape0'] == shape0)]
        if len(group) == 0:
            print('No data found for L={}, ref_prob={}'.format(
                shape0, utils.short_hex(ref_prob)))
            continue
        scalars_table = scalars_table.append(
            process_group(
                group=group,
                path=path,
                i_prob=range(min_prob, max_prob + step_prob, step_prob),
                shape0=shape0,
                ref_i_prob=ref_prob))

    scalars_table.to_csv(os.path.join(path, "scalars.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compile table of metadata.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'path', type=str, help='Path to the data')
    parser.add_argument(
        '--parameters-file', type=str, dest='parameters_file_name',
        default='summary-tables-options.yaml',
        help='YAML file containing the parameters, in the data directory')
    parser.add_argument(
        '--metadata-file', type=str, dest='metadata_file_name',
        default='file-metadata.csv',
        help='Name for the CSV file that will contain the metadata from '
             'the simulation output files')
    args = parser.parse_args()
    main(**vars(args))
