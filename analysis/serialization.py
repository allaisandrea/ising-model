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
                        'i_prob',
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
                           'parallel_count',
                           'magnetization',
                           'fourier_transform_2d'])

def make_new_observables1():
    return Observables1(
        flip_cluster_duration=list(),
        clear_flag_duration=list(),
        measure_duration=list(),
        serialize_duration=list(),
        cumulative_cluster_size=list(),
        parallel_count=list(),
        magnetization=list(),
        fourier_transform_2d=list())

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
    if version == 1 or version == 2:
        return read_metadata_v1_2(version, file)
    else:
        print("Unsupported version {}.".format(version))
        return None


def read_metadata_v1_2(version, file):
    return Metadata1(
        version=version,
        shape=read_vector('L', file),
        i_prob=read('<Q',file)[0],
        seed=read('<Q',file)[0],
        wave_numbers=read_vector('L', file),
        measure_every=read('<Q', file)[0],
        n_measure=read('<Q', file)[0],
        tag=str(bytearray(read_vector('B',file))))


def read_observables(metadata, file):
    if metadata.version == 1:
        read_observable = read_observable_v1
    elif metadata.version == 2:
        read_observable = read_observable_v2
    else:
        print("Unsupported version {}.".format(metadata.version))
        return None

    obs = make_new_observables1()
    while True:
        try:
            read_observable(file, obs)
        except struct.error:
            break
    if metadata.version == 2:
        volume = 1
        for side in metadata.shape:
            volume *= side
        for i in range(len(obs.magnetization)):
            obs.magnetization[i] = (2.0 * obs.magnetization[i]) / volume - 1.0
    return obs

def read_observable_v1(file, obs):
    obs.flip_cluster_duration.extend(read('<Q', file))
    obs.clear_flag_duration.extend(read('<Q', file))
    obs.measure_duration.extend(read('<Q', file))
    obs.serialize_duration.extend(read('<Q', file))
    obs.cumulative_cluster_size.extend(read('<Q', file))
    read('<Q', file) # n_clusters
    read('<Q', file) # representative_state
    read_vector('L', file) # stateCount
    obs.magnetization.extend(read('f', file))
    obs.fourier_transform_2d.append(read_matrix('f', file))


def read_observable_v2(file, obs):
    obs.flip_cluster_duration.extend(read('<Q', file))
    obs.clear_flag_duration.extend(read('<Q', file))
    obs.measure_duration.extend(read('<Q', file))
    obs.serialize_duration.extend(read('<Q', file))
    obs.cumulative_cluster_size.extend(read('<Q', file))
    obs.magnetization.extend(read('<Q', file))
    obs.parallel_count.extend(read('<Q', file))
    obs.fourier_transform_2d.append(read_matrix('f', file))


def read_file(file):
    metadata = read_metadata(file)
    observables = read_observables(metadata, file)
    return (metadata, observables)


