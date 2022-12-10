#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["simulate"]

import os
import sys
import time
import numpy as np
import numpy.random as rnd

from pyutils.tqdm import tqdm
from pyutils.figure import Figure
from pyutils.reservoir import LESN, Linear


def simulate(
        net, time_range, dt=1.0, x_init=None,
        innate_range=None, innate_func=None,
        innate_neuron=None, innate_every=2,
        f_in=None, f_feed=None, record_net=True, prefix=""):
    _cnt = 0
    _t = time_range[0]
    time_list = np.arange(*time_range, dt)
    pbar = tqdm(time_list, leave=False)

    if x_init is not None:
        net.reset(x_init)
    if record_net:
        rec_size = time_list.shape[0]
        rec_t = np.zeros(rec_size)
        rec_net = np.zeros((rec_size, *net.x.shape))

    for _t in pbar:
        pbar.set_description("{}t={:.0f}".format(prefix, _t))

        # step function
        u_in = np.zeros(net.x.shape)
        if f_in is not None:
            u_in += f_in(_t)
        if f_feed is not None:
            u_in += f_feed(_t, net.x)
        net.step(dt, u_in)

        # record process
        if record_net:
            rec_t[_cnt] = _t
            rec_net[_cnt] = net.x

        # innate learning process
        if (innate_range is not None) and \
                (innate_range[0] <= _t < innate_range[1]):
            if _cnt % innate_every == 0:
                x_target = innate_func(_t)
                net.innate(x_target, neuron_list=innate_neuron)
        _t += dt
        _cnt += 1
    pbar.close()
    if record_net:
        return rec_t, rec_net
