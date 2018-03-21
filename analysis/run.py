#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import subprocess
import os
import numpy as np
import signal
import uuid
import random
import argparse
from collections import namedtuple
current_process = None

def signal_handler(signum, frame):
    global current_process
    if current_process is not None:
        current_process.send_signal(signum)


def format_options(shape, prob, seed, measure_every,
                   n_measure, tag, output_path):
    options = ['--shape'] + map(str, shape) + [
               '--prob', "{:.8f}".format(prob),
               '--seed', str(seed),
               '--measure-every', str(measure_every),
               '--n-measure', str(n_measure),
               '--tag', str(tag)]
    file_name = os.path.join(output_path, uuid.uuid1().hex + '.bin')
    options += ['--output', file_name]
    return options

Parameters = namedtuple('Parameters',
                        ['prob', 'L', 'measure_every', 'n_measure'])

def read_tasks(task_file):
    tasks = list()
    with open(task_file) as in_file:
        for row in in_file:
            if row.startswith('#'):
                continue
            tokens = row.split(',')
            if len(tokens) != 4:
                continue
            tokens = [token.lstrip(' ').rstrip('\n') for token in tokens]
            tasks.append(Parameters(
                prob=(int(tokens[0], 0) / (2**32)),
                L=int(tokens[1]),
                measure_every=(1 << int(tokens[2])),
                n_measure=(1 << int(tokens[3]))))
    return tasks


def main(task_file, output_path, n_dims, executable):
    global current_process
    tasks = read_tasks(task_file)
    signal.signal(signal.SIGINT, signal_handler)
    for parameters in tasks:
        options = format_options(
            shape=([parameters.L] * n_dims),
            prob=parameters.prob,
            seed=random.randint(0, 2**32 - 1),
            measure_every=parameters.measure_every,
            n_measure=parameters.n_measure,
            tag='',
            output_path=output_path)
        command = [executable] + options
        print(' '.join(command))
        current_process = subprocess.Popen(command)
        current_process.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run simulations.')
    parser.add_argument(
        'task_file', type=str, help='File describing tasks')
    parser.add_argument(
        'output_path', type=str, help='Path where to place output')
    parser.add_argument(
        '--n-dims', type=int, default=4, help='number of dimensions')
    parser.add_argument(
        '--executable', type=str,
        default='/home/andrea/Dropbox/Documents/rg_ising/bin/run',
        help='Path to executable')
    args = parser.parse_args()
    main(**vars(args))
