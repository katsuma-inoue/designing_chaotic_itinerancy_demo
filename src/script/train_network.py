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

from pyutils.tqdm import trange
from pyutils.figure import Figure
from pyutils.interpolate import interp1d
from pyutils.reservoir import LESN

from src.library.plotter import Plotter
from src.library.simulate import simulate

parser = argparse.ArgumentParser()
parser.add_argument("--load_dir", type=str, default="../output/init")
parser.add_argument("--output_dir", type=str, default="../output")
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--washout_period", type=float, default=1000.0)
parser.add_argument("--record_offset", type=float, default=-1000.0)
parser.add_argument("--record_period", type=float, default=5000.0)
parser.add_argument("--symbol_dim", type=int, default=3)
parser.add_argument("--innate_rate", type=float, default=0.5)
parser.add_argument("--innate_offset", type=float, default=0.0)
parser.add_argument("--innate_period", type=float, default=1000.0)
parser.add_argument("--innate_epoch", type=int, default=100)
parser.add_argument("--innate_every", type=int, default=4)
parser.add_argument("--eval_num", type=int, default=5)
parser.add_argument("--save_network", action="store_true")
parser.add_argument("--save_eigen", action="store_true")
args = parser.parse_args()


def generate_random_id(sample_num, symbol_dim):
    result = np.zeros((sample_num, symbol_dim), dtype=int)
    for _i in range(sample_num):
        input_id = list(range(symbol_dim)) + [-1]
        shift = np.random.randint(1, symbol_dim + 1)
        input_id = np.roll(input_id, shift)[:-1]
        result[_i] = input_id
    return result


def create_target(w_in, net, record_range):
    net.reset(np.random.randn(args.symbol_dim, net.dim))
    net.step_while(args.dt, args.washout_period)
    rec_t, rec_net = simulate(
        net, record_range, dt=args.dt, prefix="create_target ",
        f_in=lambda _t: w_in[:args.symbol_dim] * (_t >= 0.0))
    target_func = interp1d(rec_net, x=rec_t, kind=1, axis=0)
    return (rec_t, rec_net), target_func


def train_network(w_in, net, record_range,
                  innate_func, innate_range, innate_neuron, innate_every):
    id_list = generate_random_id(1, args.symbol_dim).flatten()
    w_in_pre = w_in[id_list]
    w_in_post = w_in[:args.symbol_dim]
    net.reset(np.random.randn(args.symbol_dim, net.dim))
    net.step_while(args.dt, args.washout_period, u_in=w_in_pre)
    rec_t, rec_net = simulate(
        net, record_range, dt=args.dt, prefix="train_network ",
        innate_func=innate_func, innate_neuron=innate_neuron,
        innate_range=innate_range, innate_every=innate_every,
        f_in=lambda _t: w_in_post if _t >= 0.0 else w_in_pre)
    return (rec_t, rec_net)


def eval_network(w_in, net, record_range,
                 rec_target, innate_range, innate_neuron):
    id_list = generate_random_id(args.eval_num, args.symbol_dim)
    w_in_pre = np.zeros((args.eval_num, args.symbol_dim, net.dim))
    for _i, _id in enumerate(id_list):
        w_in_pre[_i] = w_in[_id]
    w_in_post = w_in[:args.symbol_dim]
    net.reset(np.random.randn(args.eval_num, args.symbol_dim, net.dim))
    net.step_while(args.dt, args.washout_period, u_in=w_in_pre)
    rec_t, rec_net = simulate(
        net, record_range, dt=args.dt, prefix="eval_network ",
        f_in=lambda _t: w_in_post if _t >= 0.0 else w_in_pre)
    begin_id = int((innate_range[0] - record_range[0]) / args.dt)
    end_id = int((innate_range[1] - record_range[0]) / args.dt)

    diff = rec_net.swapaxes(0, 1) - rec_target[1]
    # -> [Eval_num, Time_steps, Symbol_dim, Net_dim]
    norm = (diff[..., begin_id:end_id, :, innate_neuron] ** 2).sum(axis=-1)
    # -> [E, T, S]
    mse = norm.mean(axis=1)
    # -> [E, S]
    var = rec_target[1]
    var = (var[begin_id:end_id, :, innate_neuron]**2).sum(axis=(0, 2))
    # -> [S]
    nmse = norm.sum(axis=1) / var
    # -> [E, S]
    return (rec_t, rec_net), mse, nmse


def plot_error(error_history):
    fig = Figure()
    error_mean = np.mean(error_history, axis=1)
    error_std = np.std(error_history, axis=1)
    error_best_epoch = error_mean.sum(axis=1).argmin()
    for _mean, _std in zip(error_mean.T, error_std.T):
        fig[0].fill_std(np.arange(_mean.shape[0]), _mean, _std)
    fig[0].set_ylim([0, None])
    fig[0].line_x(error_best_epoch, ls=":", color="black")
    fig[0].set_title("best: {}".format(error_best_epoch))
    # fig[0].set_yscale("log")
    return fig


if __name__ == '__main__':
    # making target trajectory
    save_dir = "{}/{:d},{:.0f},{:.2f}".format(
        args.output_dir, args.symbol_dim, args.innate_period, args.innate_rate)
    log_dir = "{}/innate_log".format(save_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open("{}/params.json".format(save_dir), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    # loading data
    w_in = np.load("{}/w_in.npy".format(args.load_dir))
    np.save("{}/w_in.npy".format(save_dir), w_in)
    w_in = w_in[list(range(args.symbol_dim)) + [-1]]
    net = LESN.read_pickle("{}/net_init.pkl".format(args.load_dir))

    # create plotter
    graph = Plotter()
    plot_range = list(itertools.chain(
        range(0, 1), range(net.dim1, net.dim1 + 4)))
    plot_range = pd.unique([_ for _ in plot_range if _ < net.dim])

    innate_range = [args.innate_offset, args.innate_period]
    innate_neuron = range(
        net.dim1, net.dim1 + int(args.innate_rate * (net.dim2 - net.dim1)))
    innate_every = args.innate_every
    record_range = [args.record_offset, args.record_period]
    rec_target, innate_func = create_target(w_in, net, record_range)
    with open("{}/rec_target.pkl".format(log_dir), mode="wb") as f:
        joblib.dump(rec_target, f, compress=True)

    mse_best = np.inf
    mse_history = np.zeros(
        (args.innate_epoch, args.eval_num, args.symbol_dim))
    nmse_history = np.zeros(
        (args.innate_epoch, args.eval_num, args.symbol_dim))
    pbar = trange(1, args.innate_epoch + 1)
    for _epoch in pbar:
        pbar.set_description(
            "epoch:{} (best:{:.2e})".format(_epoch, mse_best))
        # training
        rec_train = train_network(w_in, net, record_range,
                                  innate_func, innate_range,
                                  innate_neuron, innate_every)
        for _id in range(args.symbol_dim):
            graph.trj_init(plot_range=plot_range)
            graph.trj_plot(rec_target[0],
                           rec_target[1][:, [_id], :],
                           color="black", ls=":", lw=1.0)
            graph.trj_plot(rec_train[0],
                           rec_train[1][:, [_id], :],
                           color="red", lw=0.2)
            graph.trj_fill(*innate_range, facecolor="cyan", alpha=0.5)
            graph.save("trj", "{}/train/{:02d}/{:03d}".format(
                log_dir, _id, _epoch))
            graph.close()

        # evaluation
        rec_eval, mse, nmse = eval_network(
            w_in, net, record_range, rec_target,
            innate_range, innate_neuron)
        mse_history[_epoch - 1], nmse_history[_epoch - 1] = mse, nmse
        for _id in range(args.symbol_dim):
            graph.trj_init(plot_range=plot_range)
            graph.trj_plot(rec_target[0], rec_target[1][:, [_id], :],
                           color="black", ls=":", lw=1.0)
            graph.trj_plot(rec_eval[0], rec_eval[1][:, :, _id, :],
                           color="red", lw=0.2)
            graph.trj_fill(*innate_range, facecolor="cyan", alpha=0.5)
            graph.save("trj", "{}/eval/{:02d}/{:03d}".format(
                log_dir, _id, _epoch))
            # graph.show()
            graph.close()

        # show error
        fig = plot_error(mse_history[:_epoch])
        fig.savefig("{}/mse_history.png".format(log_dir))
        fig.close()
        fig = plot_error(nmse_history[:_epoch])
        fig.savefig("{}/nmse_history.png".format(log_dir))
        fig.close()

        # save evaluation if best
        if mse_best > mse.sum():
            mse_best = mse.sum()
            net_tmp = copy.deepcopy(net)
            net_tmp.P = []
            net_tmp.to_pickle("{}/net_best.pkl".format(save_dir))
            np.save("{}/mse_best.npy".format(log_dir), mse)
            np.save("{}/nmse_best.npy".format(log_dir), nmse)

        # save network
        if args.save_network:
            net_tmp = copy.deepcopy(net)
            net_tmp.P = []
            net_tmp.to_pickle("{}/network/{:03d}.pkl".format(
                save_dir, _epoch))

        # eigen values
        if args.save_eigen:
            fig = Figure()
            eigvals = np.linalg.eigvals(net.w_net[net.dim1:, net.dim1:])
            ts = np.linspace(0, 2 * np.pi, 1000)
            fig[0].plot(np.cos(ts), np.sin(ts), lw=1.0, ls=":", color="black")
            fig[0].scatter(np.real(eigvals), np.imag(eigvals), s=0.8)
            fig[0].set_aspect("equal", "datalim")
            fig.savefig("{}/eig/{}.png".format(log_dir, _epoch))
            fig.close()

    # save terminal network
    net.to_pickle("{}/net_term.pkl".format(save_dir))
    np.save("{}/mse_history.npy".format(log_dir), mse_history)
    np.save("{}/nmse_history.npy".format(log_dir), nmse_history)
