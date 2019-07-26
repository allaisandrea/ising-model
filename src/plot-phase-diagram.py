#!/usr/bin/env python3
from phase_diagram_pb2 import PhaseDiagram
import sys
import struct
import numpy
import matplotlib.pyplot as plt

assert len(sys.argv) ==  2, "usage: plot-phase-diagram.py file.phd"

with open(sys.argv[1], 'rb') as in_file:
    n_read = struct.unpack('Q', in_file.read(8))[0]
    pd = PhaseDiagram()
    pd.ParseFromString(in_file.read(n_read))
    susceptibility = numpy.array(pd.susceptibility)
    n_J = pd.j_end - pd.j_begin
    n_mu = pd.mu_end - pd.mu_begin
    susceptibility = susceptibility.reshape(n_J, n_mu)
    figure, axes = plt.subplots(1, 1)
    figure.set_size_inches(4, 4)
    axes.pcolormesh(susceptibility)
    plt.savefig("foo.png")
    plt.close()



