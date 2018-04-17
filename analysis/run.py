#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import subprocess
import os
import signal
import uuid
import random
import argparse
import yaml
import time
from collections import namedtuple


processes = []
keep_running = True

def sigint_handler(signum, frame):
    global processes, keep_running
    keep_running = False
    for process in processes:
        process.send_signal(signum)


def format_options(shape, i_prob, seed, measure_every,
                   n_measure, tag, output_path, task_id):
    options = ['--shape'] + map(str, shape) + [
               '--i-prob', i_prob,
               '--seed', str(seed),
               '--measure-every', str(measure_every),
               '--n-measure', str(n_measure),
               '--tag', str(tag)]
    file_name = os.path.join(output_path, task_id + '.bin')
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


def make_commands(tasks, executable, n_dims, output_path):
    commands = []
    for parameters in tasks:
        task_id = uuid.uuid4().hex
        options = format_options(
            shape=([parameters.L] * n_dims),
            i_prob=parameters.i_prob,
            seed=random.randint(0, 2**32 - 1),
            measure_every=parameters.measure_every,
            n_measure=parameters.n_measure,
            tag='',
            output_path=output_path,
            task_id=task_id)
        commands.append((task_id, [executable] + options))
    return commands


def main(task_file, output_path, n_dims, executable, n_processes, poweroff):
    global processes, keep_running
    signal.signal(signal.SIGINT, sigint_handler)
    tasks = read_tasks(task_file)
    commands = make_commands(tasks, executable, n_dims, output_path)
    log_files = []
    running_commands = []
    while keep_running and (len(commands) > 0 or len(processes) > 0):
        # Check for completion
        new_processes = []
        new_log_files = []
        new_running_commands = []
        for i, process in enumerate(processes):
            return_code = process.poll()
            if return_code is None:
                new_processes.append(process)
                new_log_files.append(log_files[i])
                new_running_commands.append(running_commands[i])
            else:
                task, command = running_commands[i]
                print("Task {} complete with return code {}:".format(
                    task, return_code))
                print(' '.join(command))
                log_files[i].close()
        processes = new_processes
        log_files = new_log_files
        running_commands = new_running_commands

        # Enqueue new jobs
        while len(commands) > 0 and len(processes) < n_processes:
            task_id, command = commands.pop()
            print('Start task {}:'.format(task_id))
            print(' '.join(command))
            log_files.append(
                open(os.path.join(output_path, task_id + ".log"), 'w'))
            processes.append(subprocess.Popen(
                command,
                stdout=log_files[-1],
                stderr=log_files[-1]))
            running_commands.append((task_id, command))

        # Wait
        time.sleep(120)


    if poweroff and keep_running:
        subprocess.call(['sudo', 'poweroff'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run simulations.')
    parser.add_argument(
        'task_file', type=str, help='YAML file describing tasks')
    parser.add_argument(
        'output_path', type=str, help='Path where to place output')
    parser.add_argument(
        '--n-dims', type=int, default=4, help='number of dimensions')
    parser.add_argument(
        '--executable', type=str, default='../build/run',
        help='Path to executable')
    parser.add_argument(
        '--n-processes', type=int, default=1, help='number of processes')
    parser.add_argument(
        '--poweroff', action='store_true', default=False,
        help='Power off at the end')
    args = parser.parse_args()
    main(**vars(args))
