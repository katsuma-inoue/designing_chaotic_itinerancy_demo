#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import copy
import joblib
import argparse
import itertools
import numpy as np
import pandas as pd

sys.path.append(".")

from pyutils.figure import Figure
from pyutils.reservoir import LESN, Softmax

from src.library.plotter import Plotter
from src.library.simulate import simulate
from src.library.utils import random_chain

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as clr

parser = argparse.ArgumentParser()
parser.add_argument("load_dir", type=str)
parser.add_argument("--use_cache", action="store_true")
parser.add_argument("--w_in_path", type=str, default="w_in.npy")
parser.add_argument("--net_path", type=str, default="net_term.pkl")
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--washout_period", type=float, default=1000.0)
parser.add_argument("--noise_gain", type=float, default=None)
parser.add_argument("--symbol_dim", type=int, default=3)
parser.add_argument("--symbol_list", type=int, nargs="+", default=None)
parser.add_argument("--sample_num", type=int, default=30)
parser.add_argument("--sample_offset", type=float, default=0.0)
parser.add_argument("--sample_period", type=float, default=10000.0)
parser.add_argument("--sample_pert", type=float, default=None)
parser.add_argument("--switch_period", type=int, nargs="+", default=[3000])
parser.add_argument("--prob_pattern", type=str,
                    choices=["cyclic", "uniform", "branched",
                             "history_dependent"],
                    default="uniform")
parser.add_argument("--eval_num", type=int, default=10)
parser.add_argument("--eval_period", type=float, default=50000.0)
parser.add_argument("--eval_pert", type=float, default=1e-3)
parser.add_argument("--tol", type=float, default=1e-1)
parser.add_argument("--n_jobs", type=int, default=1)
parser.add_argument('--solver',
                    choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                    default="lbfgs")
parser.add_argument("--multi_class", type=str, default="multinomial")
args = parser.parse_args()


def symbol_generator():
    print("pattern: {}".format(args.prob_pattern))
    if args.prob_pattern == "cyclic":
        symbol_list = []
        if args.symbol_list is None:
            rotate_list = np.arange(args.symbol_dim)
        else:
            rotate_list = np.array(args.symbol_list)
        for _ in range(args.sample_num):
            symbol_list.append(rotate_list[
                np.arange(_, _ + 100) % len(rotate_list)])
        symbol_list = np.array(symbol_list).T
    else:
        if args.prob_pattern == "uniform":
            prob = np.ones([args.symbol_dim, args.symbol_dim]) - \
                np.eye(args.symbol_dim)
        if args.prob_pattern == "branched":
            prob = np.ones([args.symbol_dim, args.symbol_dim])
            prob[1:, 1:] = 0.0
            prob[0, 0] = 0.0
        if args.prob_pattern == "history_dependent":
            prob = np.zeros([args.symbol_dim * 2 - 1, args.symbol_dim * 2 - 1])
            prob[0, 1:args.symbol_dim] = 1.0
            for _ in range(args.symbol_dim - 1):
                _i = 1 + _
                _j = args.symbol_dim + (_ + 1) % (args.symbol_dim - 1)
                prob[_i, _j] = 1.0
            prob[(1 - args.symbol_dim):, 0] = 1.0
        print("target transition prob: ")
        print(np.einsum("ij,i->ij", prob, 1 / prob.sum(axis=1)))
        symbol_list = random_chain(
            args.symbol_dim, (100, args.sample_num),
            init_id=None, prob=prob).astype(int)
        if args.symbol_list is not None:
            symbol_list = np.array(args.symbol_list)[np.array(symbol_list)]
    return symbol_list


def sample_network(w_in, net):
    if args.sample_pert is None:
        net.reset(np.random.randn(args.sample_num, net.dim))
        net.step_while(args.dt, args.washout_period)
    else:
        net.reset(np.random.randn(1, net.dim))
        net.step_while(args.dt, args.washout_period)
        pert = args.sample_pert * np.random.randn(args.sample_num, net.dim)
        pert[:, :net.dim1] = 0.0
        net.reset(net.x + pert)

    symbol_list = symbol_generator()
    input_list = []
    for _symbol in symbol_list.T:
        symbol_repeat = np.repeat(
            _symbol, np.array(args.switch_period)[
                _symbol % len(args.switch_period)])
        input_list.append(symbol_repeat)
    length = len(min(input_list, key=lambda arr: len(arr)))
    input_list = [_[:length] for _ in input_list]
    input_list = np.array(input_list).T

    if args.prob_pattern == "history_dependent":
        input_list[input_list >= args.symbol_dim] -= args.symbol_dim - 1
        # print(symbol_list)

    def f_in(_t):
        return w_in[input_list[int(_t) % input_list.shape[0]]]

    record_range = [0, args.sample_period]
    rec_t, rec_net = simulate(
        net, record_range, dt=args.dt, f_in=f_in, prefix="sample_newtork ")
    begin_id = int(args.sample_offset / args.dt)
    end_id = int(args.sample_period / args.dt)
    x_train = rec_net[begin_id:end_id - 1].reshape((-1, net.dim))
    y_train = input_list[begin_id + 1:end_id].reshape(-1)
    return (rec_t, rec_net), (x_train, y_train)


def eval_network(w_in, net, w_feed):
    if args.eval_pert is None:
        net.reset(np.random.randn(args.eval_num, net.dim))
        net.step_while(args.dt, args.washout_period)
    else:
        net.reset(np.random.randn(1, net.dim))
        net.step_while(args.dt, args.washout_period)
        pert = args.eval_pert * np.random.randn(args.eval_num, net.dim)
        pert[:, :net.dim1] = 0.0
        x_init = np.concatenate([net.x, net.x + pert], axis=0)
        net.reset(x_init)

    def f_feed(_t, _x):
        if _t < 0.0:
            return 0.0
        else:
            return w_in[w_feed.predict(_x)]

    record_range = [0, args.eval_period]
    rec_t, rec_net = simulate(
        net, record_range, dt=args.dt, f_feed=f_feed, prefix="eval_network ")

    if args.eval_pert:
        d_init = x_init[1:] - x_init[0]
        d_eval = rec_net[:, 1:] - rec_net[:, :1]

        d_pre = np.linalg.norm(d_init, axis=1)
        d_post = np.linalg.norm(d_eval, axis=2)
        d_ratio = np.log(d_post / d_pre)
    else:
        d_ratio = None

    return (rec_t, rec_net), d_ratio


if __name__ == '__main__':
    # loading data
    w_in = np.load("{}/{}".format(args.load_dir, args.w_in_path))
    w_in = w_in[list(range(args.symbol_dim)) + [-1]]
    net = LESN.read_pickle("{}/{}".format(args.load_dir, args.net_path))
    if args.noise_gain is not None:
        noise_gain = np.ones(net.dim) * args.noise_gain
        noise_gain[:net.dim1] = 0.0
        net.noise_gain = noise_gain

    # create save_dir
    if args.prob_pattern == "cyclic":
        if args.symbol_list is None:
            args.symbol_list = list(map(int, np.arange(args.symbol_dim)))
        suffix = "-{}".format(
            "-".join(map(str, args.symbol_list)))
    else:
        suffix = ""
    save_dir = "{}/{}{},{}".format(
        args.load_dir, args.prob_pattern, suffix,
        "-".join(map(str, args.switch_period)))
    if args.noise_gain is not None:
        save_dir += ",{:.2e}".format(args.noise_gain)
    print("save to {}".format(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    # create plotter
    graph = Plotter()
    plot_range = list(itertools.chain(
        range(0, 1), range(net.dim1, net.dim1 + 4)))
    plot_range = pd.unique([_ for _ in plot_range if _ < net.dim])

    # train or retrieve w_feed
    if args.use_cache:
        with open("{}/w_feed.pkl".format(save_dir), mode="rb") as f:
            w_feed = joblib.load(f)
    else:
        with open("{}/params.json".format(save_dir), mode="w") as f:
            json.dump(args.__dict__, f, indent=4)
        rec_sample, dataset = sample_network(w_in, net)
        w_feed = Softmax(verbose=1, multi_class=args.multi_class,
                         solver=args.solver, tol=args.tol, n_jobs=args.n_jobs)
        w_feed.fit(*dataset)
        with open("{}/w_feed.pkl".format(save_dir), mode="wb") as f:
            joblib.dump(w_feed, f, compress=True)
        net.P = []
        with open("{}/setup.pkl".format(save_dir), mode="wb") as f:
            joblib.dump([w_in, net, w_feed], f, compress=True)

    rec_eval, d_ratio = eval_network(w_in, net, w_feed)
    # graph.trj_init(plot_range=plot_range)
    # graph.trj_plot(*rec_eval, ls="-", lw=1.0)
    graph.feed_init(w_feed)
    graph.feed_plot(*rec_eval, ls="-", lw=1.0)
    graph.symbol_init(w_feed)
    df_transition, df_period = graph.symbol_plot(*rec_eval)

    # save
    graph.save("feed", "{}/symbol_dynamics".format(save_dir))
    graph.save("symbol", "{}/symbol_distribution".format(save_dir))
    df_transition.to_csv("{}/transition.csv".format(save_dir))
    df_period.to_csv("{}/period.csv".format(save_dir))
    if d_ratio is not None:
        fig = Figure()
        fig[0].fill_std(
            rec_eval[0], np.mean(d_ratio, axis=1),
            np.std(d_ratio, axis=1))
        fig.savefig("{}/distance_dynamics.png".format(save_dir))
