#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["Timer", "print", "get_func_name", "random_chain"]

import os
import sys
import time
import inspect
import numpy as np
import pandas as pd
import builtins as __builtin__

from pyutils.random import markov_chain
from pyutils.interpolate import interp1d, curve_normalize


class Timer(object):
    def __init__(self):
        self.begin = time.time()

    def tic(self):
        self.start = time.time()

    def toc(self):
        return time.time() - self.start


begin = time.time()


def print_with_time(*args, prefix="\x1b[2K", **kwargs):
    __builtin__.print("{}[{:8.3f}]".format(
        prefix, time.time() - begin), *args, **kwargs)


print = print_with_time


def get_func_name(depth=1):
    return inspect.stack()[depth][3]


def random_chain(dim, shape, init_id=None, prob=None):
    id_list = []
    if prob is None:
        prob = 1 - np.eye(dim)
    for _ in range(shape[1]):
        id_list.append(markov_chain(prob, shape[0], init_state=init_id))
    return np.array(id_list).T
