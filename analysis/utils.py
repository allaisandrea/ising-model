#! /usr/bin/env python
from __future__ import print_function
from __future__ import division
import numpy


def autocorrelation(v, n):
    if n > 0:
        v1 = v[n:] - numpy.mean(v[n:])
        v2 = v[:-n] - numpy.mean(v[:-n])
    else:
        v1 = v
        v2 = v
    return numpy.dot(v1, v2) / (v1.size * numpy.std(v1) * numpy.std(v2))


def cross_validate(f, v, n):
    v1 = v[:((v.shape[0] // n) * n)]
    v1 = v1.reshape((n, v1.shape[0] // n) + v1.shape[1:])
    v1 = numpy.stack([f(x) for x in v1])
    v1 = numpy.array([f(v), numpy.std(v1, axis=0) / numpy.sqrt(n)])
    v1 = numpy.transpose(v1, range(1, len(v1.shape)) + [0])
    return v1