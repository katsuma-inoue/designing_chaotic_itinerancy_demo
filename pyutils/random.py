#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["roulette_choice", "shuffle", "markov_chain"]

import numpy as np
import numpy.random as rnd
import bisect
import random


def roulette_choice(w):
    '''
    Choosing index with probability array
    Generating accumulate weight with numpy.cumsum()
    '''
    tot = np.cumsum(w)
    return bisect.bisect_left(tot, random.random() * tot[-1])


def shuffle(*args, seed=None):
    '''
    Shuffling multiple arrays to same order
    '''
    random.seed(seed)
    if len(args) is 1:
        random.shuffle(*args)
        return args[0]
    c = list(zip(*args))
    random.shuffle(c)
    return tuple(map(list, zip(*c)))


def markov_chain(prob, N, init_state=None):
    '''
    Generating N-step markov chain with probablistic matrix

    Args:
        prob (np.ndarray) : probablistics matrix
        N (int) : the number of elements in generated array
        init_state (list) : init states (default=None)
    Returns:
        list: N-step array
    '''
    series = []
    prob = np.array(prob)
    state = rnd.randint(prob.shape[0]) if init_state is None else init_state
    series.append(state)
    for _ in range(N - 1):
        state = roulette_choice(prob[state])
        series.append(state)
    return series
