#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import math
import json
import joblib
import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.append(".")

from pyutils.chaos import lorenz, runge_kutta
from pyutils.figure import Figure

from src.library.plotter import Plotter
from src.library.simulate import simulate
from src.library.utils import random_chain

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("load_dir", type=str)
parser.add_argument("--use_cache", action="store_true")
parser.add_argument("--w_in_path", type=str, default="w_in.npy")
parser.add_argument("--net_path", type=str, default="net_term.pkl")
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--washout_period", type=float, default=1000.0)
parser.add_argument("--func_period", type=int, nargs="+", default=[1000])
parser.add_argument("--func_dim", type=int, default=3)
parser.add_argument("--func_num_max", type=int, default=3)
parser.add_argument("--func_freq", type=float, default=4.0)
parser.add_argument("--sample_num", type=int, default=30)
parser.add_argument("--sample_offset", type=int, default=0)
parser.add_argument("--sample_washout", type=int, default=0)
parser.add_argument("--sample_period", type=float, default=10000.0)
parser.add_argument("--eval_num", type=int, default=10)
parser.add_argument("--eval_period", type=float, default=10000.0)
parser.add_argument("--eval_pert", type=float, default=None)
parser.add_argument("--alpha", type=float, default=1.0)
args = parser.parse_args()


def plot_2d(*_args):
    fig = Figure()
    for data in _args:
        fig[0].plot(data[:, 0], data[:, 1], lw=0.5)
    fig[0].set_aspect("equal", "datalim")
    return fig


def plot_3d(*_args):
    fig = Figure()
    ax = fig[0].convert_3d()
    for data in _args:
        ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5)
    return fig


def plot_ts(*_args):
    fig = Figure()
    fig.create_grid((_args[0].shape[1], 1), hspace=0)
    for data in _args:
        for _ in range(data.shape[1]):
            fig[_].plot(data[:, _])
    return fig


def circle_offset(index, func_dim, radius=1.0):
    return radius * np.array([
        np.cos(2 * np.pi * index / func_dim),
        np.sin(2 * np.pi * index / func_dim)])


def lissajous_function(
        freq, offset=np.array([0, 0]), radius=1.0):
    def func(_t):
        return radius * np.array([
            np.sin(2 * np.pi * freq[0] * _t),
            np.sin(2 * np.pi * freq[1] * _t)]).T + offset
    return func


def create_target():
    target_list = []
    for _i in range(args.func_dim):
        freq = args.func_freq * np.array([
            _i % args.func_num_max + 2,
            (_i + 1) % args.func_num_max + 2])
        radius = math.sqrt(2) / math.sin(math.pi / args.func_dim)
        offset = circle_offset(_i, args.func_dim, radius=radius)
        func = lissajous_function(freq, offset)
        func_len = args.func_period[_i % len(args.func_period)]
        target_list.append(func(np.linspace(0, 1, func_len)))
    # target_list = [np.concatenate([
    #     _target, np.tile(_target[-1:], (args.sample_offset, 1))], axis=0)
    #     for _i, _target in enumerate(target_list)]
    return target_list


def symbol_generator(classes, dim):
    symbol_list = random_chain(len(classes), (100, dim)).astype(int)
    symbol_list = classes[symbol_list]
    input_list = []
    for _symbol in symbol_list.T:
        symbol_repeat = np.repeat(
            _symbol, np.array(args.func_period)[
                _symbol % len(args.func_period)])
        input_list.append(symbol_repeat)
    length = len(min(input_list, key=lambda arr: len(arr)))
    input_list = [_[:length] for _ in input_list]
    input_list = np.array(input_list).T
    return input_list


def sample_network(w_in, net, target_list):
    net.reset(np.random.randn(args.sample_num, net.dim))
    net.step_while(args.dt, args.washout_period)

    classes = np.arange(len(target_list))
    symbol_list = symbol_generator(classes, args.sample_num)

    def f_feed(_t, _x):
        return w_in[symbol_list[int(_t)]]

    record_range = [0, args.sample_period]
    rec_t, rec_net = simulate(
        net, record_range, dt=args.dt, f_feed=f_feed, prefix="sample_newtork ")

    x_list, y_list = [], []
    for _i, _x in enumerate(np.rollaxis(rec_net[0:, :], 1)):
        label_list = symbol_list[:rec_t.shape[0], _i]
        key_list, border_list = [], []
        for _key, _group in itertools.groupby(label_list):
            key_list.append(_key)
            border_list.append(len(list(_group)))
        border_list = np.cumsum([0] + border_list)
        generator = list(zip(key_list, border_list[:-1], border_list[1:]))
        for _key, _begin, _end in generator[args.sample_washout:]:
            if _key == -1:
                continue
            target = target_list[_key % len(target_list)]
            _begin_id = _begin + args.sample_offset
            _end_id = min(_end, _begin + len(target))
            x_list.append(_x[_begin_id:_end_id])
            y_list.append(target[:len(x_list[-1])])
    X = np.concatenate(x_list, axis=0)
    Y = np.concatenate(y_list, axis=0)
    return (rec_t, rec_net), X, Y


def eval_network(w_in, net, w_out):
    if args.eval_pert is None:
        net.reset(np.random.randn(args.eval_num, net.dim))
        net.step_while(args.dt, args.washout_period)
    else:
        net.reset(np.random.randn(net.dim))
        net.step_while(args.dt, args.washout_period)
        pert = args.eval_pert * np.random.randn(args.eval_num, net.dim)
        pert[:, :net.dim1] = 0.0
        net.reset(pert + net.x)

    classes = np.array(w_out.classes_)
    symbol_list = symbol_generator(classes, args.eval_num)

    def f_feed(_t, _x):
        return w_in[symbol_list[int(_t)]]

    record_range = [0, args.eval_period]
    rec_t, rec_net = simulate(
        net, record_range, dt=args.dt, f_feed=f_feed, prefix="eval_newtork ")
    rec_out = np.array([w_out.predict(_x) for _x in rec_net])

    fig1 = Figure()
    fig1.create_grid((rec_out.shape[2], 1), hspace=0)
    fig2 = Figure()
    for _out in np.rollaxis(rec_out, 1):
        for _ in range(_out.shape[1]):
            fig1[_].plot(rec_t, _out[:, _])
            if _ < _out.shape[1] - 1:
                fig1[_].set_xticklabels([])
        fig2[0].plot(_out[:, 0], _out[:, 1])
        fig2[0].set_aspect("equal", "datalim")
    return (rec_t, rec_net), fig1, fig2


if __name__ == '__main__':
    # loading data
    with open("{}/{}".format(
            args.load_dir, args.net_path), mode="rb") as f:
        net = joblib.load(f)
    w_in = np.load("{}/{}".format(args.load_dir, args.w_in_path))

    # create save_dir
    save_dir = "{}/readout,{}".format(
        args.load_dir, "-".join(map(str, args.func_period)))
    print("save to {}".format(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    # create plotter
    graph = Plotter()
    plot_range = list(itertools.chain(
        range(0, 1), range(net.dim1, net.dim1 + 4)))
    plot_range = pd.unique([_ for _ in plot_range if _ < net.dim])

    # train or retrieve w_out
    if args.use_cache:
        with open("{}/w_out.pkl".format(save_dir), mode="rb") as f:
            w_out = joblib.load(f)
    else:
        with open("{}/params.json".format(save_dir), mode="w") as f:
            json.dump(args.__dict__, f, indent=4)
        target_list = create_target()
        fig1 = plot_ts(*target_list)
        fig2 = plot_2d(*target_list)
        rec_sample, X, Y = sample_network(w_in, net, target_list)
        w_out = Ridge(alpha=args.alpha)
        w_out.fit(X, Y)
        w_out.classes_ = np.arange(len(target_list))
        net.P = []
        with open("{}/w_out.pkl".format(save_dir), mode="wb") as f:
            joblib.dump(w_out, f, compress=True)

    rec_eval, fig1, fig2 = eval_network(w_in, net, w_out)
    # graph.trj_init(plot_range=plot_range)
    # graph.trj_plot(*rec_eval, ls="-", lw=1.0)

    # save
    fig1.savefig("{}/out_1d.png".format(save_dir))
    fig2.savefig("{}/out_2d.png".format(save_dir))
