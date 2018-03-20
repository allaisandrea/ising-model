#!/usr/bin/env python

import subprocess
import numpy as np
import signal
import uuid

current_process = None

def signal_handler(signum, frame):
    global current_process
    if current_process is not None:
        current_process.send_signal(signum)


def format_options(shape, prob, seed, measure_every, n_measure, tag):
    options = ['--shape'] + map(str, shape) + [ 
               '--prob', "{:.8f}".format(prob),
               '--seed', str(seed),
               '--measure-every', str(measure_every), 
               '--n-measure', str(n_measure),
               '--tag', str(tag)]
    file_name = uuid.uuid1().hex + '.bin'
    options += ['--output', file_name]
    return options
    
def main():
    global current_process
    signal.signal(signal.SIGINT, signal_handler)
    executable = '../bin/run'
    prob = 0.645
    for L, measure_every, n_measure in [
        #(8,            64,  16 * 1024),
        #(12,           64,  16 * 1024),
        #(16,          128,  16 * 1024),
        #(24,          128,  16 * 1024),
        #(32,          256, 1024 * 1024),
        #(48,          256,  1024 * 1024),
        #(64,          512,  1024 * 1024),
        (128,    2 * 1024,  1024 * 1024),
        #(256,    8 * 1024,   32 * 1024),
        #(512,   32 * 1024,    8 * 1024),
        #(1024, 512 * 1024,   4 * 1024),
        ]:
        options = format_options(
            shape=[L, L, L],
            prob=prob,
            seed=6,
            measure_every=measure_every,
            n_measure=n_measure,
            tag='autocorrelation-1')
        command = [executable] + options
        print(' '.join(command))
        current_process = subprocess.Popen(command)
        current_process.wait()

if __name__ == '__main__':
    main()