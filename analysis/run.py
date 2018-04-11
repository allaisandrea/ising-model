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
import yaml
from collections import namedtuple


current_process = None
keep_running = True

def sigint_handler(signum, frame):
    global current_process, keep_running
    keep_running = False
    if current_process is not None:
        current_process.send_signal(signum)


def format_options(shape, i_prob, seed, measure_every,
                   n_measure, tag, output_path):
    options = ['--shape'] + map(str, shape) + [
               '--i-prob', i_prob,
               '--seed', str(seed),
               '--measure-every', str(measure_every),
               '--n-measure', str(n_measure),
               '--tag', str(tag)]
    file_name = os.path.join(output_path, uuid.uuid1().hex + '.bin')
    options += ['--output', file_name]
    return options

Parameters = namedtuple('Parameters',
                        ['i_prob', 'L', 'measure_every', 'n_measure'])

def read_tasks(task_file):
    with open(task_file) as in_file:
        tasks = yaml.load(in_file)
    tasks = [
        Parameters(
            i_prob=tokens[0],
            L=int(tokens[1]),
            measure_every=(1 << int(tokens[2])),
            n_measure=(1 << int(tokens[3])))
        for tokens in tasks]
    return tasks


def main(task_file, output_path, n_dims, executable):
    global current_process, keep_running
    tasks = read_tasks(task_file)
    signal.signal(signal.SIGINT, sigint_handler)
    for parameters in tasks:
        if not keep_running:
            break
        options = format_options(
            shape=([parameters.L] * n_dims),
            i_prob=parameters.i_prob,
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
        'task_file', type=str, help='YAML file describing tasks')
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
