#! /usr/bin/env python
from __future__ import print_function
from __future__ import division
import struct
from collections import namedtuple
import numpy
import os
import glob
import pandas as pd


Metadata1 = namedtuple('Metadata1',
                       ['version',
                        'shape',
                        'prob',
                        'seed',
                        'wave_numbers',
                        'measure_every',
                        'n_measure',
                        'tag'])


Observables1 = namedtuple('Observables1',
                          ['flip_cluster_duration',
                           'clear_flag_duration',
                           'measure_duration',
                           'serialize_duration',
                           'cumulative_cluster_size',
                           'n_clusters',
                           'magnetization',
                           'fourier_transform_2d'])


def read(format, file):
    size = struct.calcsize(format)
    return struct.unpack(format, file.read(size))


def read_vector(type, file):
    n, = read('<Q', file)
    return read('<{}{}'.format(n, type), file)


def read_matrix(type, file):
    n1, = read('<Q', file)
    n2, = read('<Q', file)
    res = numpy.array(read('<{}{}'.format(n1 * n2, type), file))
    return res.reshape(n1, n2)


def read_metadata(file):
    version, = read('<Q', file)
    if version == 1:
        return read_metadata_v1(file)
    else:
        print("Unsupported version {}.".format(version))
        return None


def read_metadata_v1(file):
    return Metadata1(
        version=1,
        shape=read_vector('L', file),
        prob=read('d',file)[0],
        seed=read('<Q',file)[0],
        wave_numbers=read_vector('L', file),
        measure_every=read('<Q', file)[0],
        n_measure=read('<Q', file)[0],
        tag=str(bytearray(read_vector('B',file))))


def read_observables(version, file):
    if version == 1:
        return read_observables_v1(file)
    else:
        print("Unsupported version {}.".format(version))
        return None


def read_observables_v1(file):
    obs = Observables1(
        flip_cluster_duration=list(),
        clear_flag_duration=list(),
        measure_duration=list(),
        serialize_duration=list(),
        cumulative_cluster_size=list(),
        n_clusters=list(),
        magnetization=list(),
        fourier_transform_2d=list())
    while True:
        try:
            read_observable_v1(file, obs)
        except struct.error:
            break
    return obs


def convert_observables_to_numpy(obs):
    return Observables1(
        flip_cluster_duration   = numpy.array(obs.flip_cluster_duration  ),
        clear_flag_duration     = numpy.array(obs.clear_flag_duration    ),
        measure_duration        = numpy.array(obs.measure_duration       ),
        serialize_duration      = numpy.array(obs.serialize_duration     ),
        cumulative_cluster_size = numpy.array(obs.cumulative_cluster_size),
        n_clusters              = numpy.array(obs.n_clusters             ),
        magnetization           = numpy.array(obs.magnetization          ),
        fourier_transform_2d    = None   )

def read_observable_v1(file, obs):
    obs.flip_cluster_duration.extend(read('<Q', file))
    obs.clear_flag_duration.extend(read('<Q', file))
    obs.measure_duration.extend(read('<Q', file))
    obs.serialize_duration.extend(read('<Q', file))
    obs.cumulative_cluster_size.extend(read('<Q', file))
    obs.n_clusters.extend(read('<Q', file))
    read('<Q', file) # representative_state
    read_vector('L', file) # stateCount
    obs.magnetization.extend(read('f', file))
    read_matrix('f', file) # fourier_transform_2d


def read_file(file):
    metadata = read_metadata(file)
    observables = read_observables(metadata.version, file)
    return (metadata, observables)


def get_shape_field_names():
    return ("shape0", "shape1",  "shape2", "shape3", "shape4")


def get_file_metadata_table(input_path):
    file_names = glob.glob(os.path.join(input_path, "*.bin"))
    metadata_fields =list(Metadata1._fields)
    metadata_fields.remove('wave_numbers')
    metadata_fields.remove('shape')
    shape_fiels = get_shape_field_names()
    metadata_list = list()
    for file_name in file_names:
        with open(file_name) as in_file:
            metadata = read_metadata(in_file)
        statinfo = os.stat(file_name)
        base_name = os.path.basename(file_name)
        shape = [0] * 5
        for i, shape_i in enumerate(metadata.shape):
            shape[i] = shape_i
        metadata = tuple(getattr(metadata, tag) for tag in metadata_fields)
        metadata_list.append((base_name, statinfo.st_size) +
                             tuple(shape) + metadata)

    column_names = ("file_name", "file_size")
    column_names += shape_fiels + tuple(metadata_fields)
    return pd.DataFrame(metadata_list, columns=column_names)


def find_duplicates(file_metadata_table):
    unique_field_names = list(get_shape_field_names() + ("prob", "seed"))
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


def get_group_unique_field_names():
    return get_shape_field_names() + ("prob", "measure_every")


def get_metadata_table(file_metadata_table):
    field_names = list(get_group_unique_field_names())
    table = file_metadata_table[field_names]
    return table.groupby(field_names[:-1])['measure_every'].max()
