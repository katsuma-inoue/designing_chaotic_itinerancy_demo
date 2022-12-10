#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import joblib
import argparse
import warnings
import itertools
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LogisticRegression

sys.path.append(".")

from pyutils.figure import Figure
from pyutils.interpolate import interp1d
from pyutils.reservoir import LESN, DESN, Linear

from src.library.plotter import Plotter
from src.library.simulate import simulate
from src.library.utils import random_chain

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
# experiment configurations
parser.add_argument("--output_dir", type=str, default="../output/")
parser.add_argument("--seed", type=int, default=None)
# input parameters
parser.add_argument("--symbol_dim", type=int, default=5)
parser.add_argument("--w_in_amp", type=float, default=0.01)
# connection parameter
parser.add_argument("--con_alpha", type=float, default=0)
parser.add_argument("--con_tau", type=float, default=50.0)
parser.add_argument("--con_amp", type=float, default=1.0)
# network paramters
parser.add_argument("--net_mode", type=str, default="normal")
parser.add_argument("--net_tau", type=float, default=10.0)
parser.add_argument("--net_alpha", type=float, default=1.0)
parser.add_argument("--net_input_dim", type=int, default=500)
parser.add_argument("--net_input_g", type=float, default=0.9)
parser.add_argument("--net_input_scale", type=float, default=1.0)
parser.add_argument("--net_chaos_dim", type=int, default=1000)
parser.add_argument("--net_chaos_g", type=float, default=1.5)
parser.add_argument("--net_chaos_scale", type=float, default=0.1)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--precision", type=float, default=0.99)
parser.add_argument("--threshold", type=float, nargs="+", default=[0, 10000.0])
parser.add_argument("--eval_period", type=float, default=10000.0)
parser.add_argument("--washout_period", type=float, default=1000.0)
args = parser.parse_args()

rnd = np.random.RandomState(seed=args.seed)


def create_input_reservoir():
    net = LESN(
        args.net_input_dim, args.net_tau,
        g=args.net_input_g, scale=args.net_input_scale,
        rnd=rnd, mode=args.net_mode)
    w_in = rnd.uniform(
        size=(args.symbol_dim + 1, net.dim), low=-1.0, high=1.0)
    w_in *= args.w_in_amp
    w_in[-1] = 0.0

    def transient(_x, tau):
        res = _x * np.exp(-(_x / tau)**2 / 2)
        res[np.isnan(res)] = 0.0
        return res

    def generate_target(symbol_list, switch_period):
        y_desire = np.full((
            symbol_list.shape[0] * switch_period,
            args.symbol_dim), np.nan)
        for _id in range(args.symbol_dim):
            on_list = np.arange(symbol_list.shape[0])[symbol_list == _id]
            on_list = np.append(on_list, symbol_list.shape[0])
            on_list *= switch_period
            for _pre, _post in zip(on_list[:-1], on_list[1:]):
                y_desire[_pre:_post, _id] = np.arange(_post - _pre)
        return transient(y_desire, tau=args.con_tau)

    def run_simulate(symbol_list, switch_period):
        input_id_list = symbol_list.repeat(switch_period)
        return simulate(
            net, [0, input_id_list.shape[0]], dt=args.dt,
            f_in=lambda t: w_in[input_id_list[int(t)]],
            prefix="create_input_reservoir ")

    net.reset(rnd.randn(net.dim))
    net.step_while(args.dt, args.washout_period)
    symbol_list = np.array(sum(itertools.permutations(
        np.arange(-1, args.symbol_dim), 2), ()))
    symbol_list = np.concatenate([
        symbol_list[-args.symbol_dim * 2:],
        symbol_list[:-args.symbol_dim * 2]])
    rec_t, rec_net = run_simulate(symbol_list, 500)
    x_train = rec_net
    y_train = generate_target(symbol_list, 500)
    w_out = Ridge(alpha=args.con_alpha, fit_intercept=False)
    w_out.fit(x_train, y_train)

    # print(symbol_list)
    # print(w_out.coef_.max())
    # fig = Figure()
    # fig.create_grid((args.symbol_dim, 1), hspace=0.0)
    # y_predict = w_out.predict(x_train)
    # for _id in range(args.symbol_dim):
    #     fig[_id].plot(rec_t, y_train[:, _id],
    #                   color="black", ls=":")
    #     fig[_id].plot(rec_t, y_predict[:, _id])
    # Figure.show()
    # fig.close()

    net.reset(rnd.randn(net.dim))
    net.step_while(args.dt, args.washout_period)
    symbol_list = random_chain(
        args.symbol_dim + 1, (args.symbol_dim * 4, 1)).T[0] - 1
    rec_t, rec_net = run_simulate(symbol_list, 500)
    x_eval = rec_net
    y_eval = generate_target(symbol_list, 500)
    y_predict = w_out.predict(x_eval)

    # fig = Figure()
    # fig.create_grid((args.symbol_dim, 1), hspace=0.0)
    score_list = []
    for _id in range(args.symbol_dim):
        # fig[_id].plot(rec_t, y_eval[:, _id], color="black", ls=":")
        # fig[_id].plot(rec_t, y_predict[:, _id])
        score_list.append([
            pearsonr(y_eval[:, _id], y_predict[:, _id])[0]**2])
    score_list = np.array(score_list)
    # Figure.show()
    # fig.close()
    print("score: {:.4e}±{:.4e}".format(
        score_list.mean(), score_list.std()))
    score = score_list.mean()
    if score < args.precision or np.isnan(score):
        print("call create_input_reservoir() again!")
        return create_input_reservoir()
    return w_in, net, w_out.coef_


def create_network():
    input_dim = args.net_input_dim
    net_dim = args.net_input_dim + args.net_chaos_dim
    # creating input_reservoir
    w_in, net_input, w_con = create_input_reservoir()
    w_in = np.concatenate([w_in, np.zeros(
        (w_in.shape[0], net_dim - input_dim))], axis=1)
    net_chaos = LESN(
        args.net_chaos_dim, args.net_tau,
        scale=args.net_chaos_scale, g=args.net_chaos_g,
        rnd=rnd, mode=args.net_mode)
    # concatenating two network
    net = LESN.concatenate([net_input, net_chaos])
    net.w_net[input_dim:net_dim, :input_dim] = \
        rnd.randn(net_dim - input_dim, w_con.shape[0]).dot(w_con)
    net.w_net[input_dim:net_dim, :input_dim] *= args.con_amp
    net.dim1, net.dim2 = input_dim, net_dim
    net.reset_rls(
        alpha=args.net_alpha, mu=1.0,
        pre_node=range(net.dim1, net.dim2),
        post_node=range(net.dim1, net.dim2))
    return w_in, net


def eval_network(w_in, net, symbol_id):
    eval_num = args.symbol_dim * 2
    net.reset(np.random.randn(eval_num, net.dim))
    input_id = np.arange(eval_num) % args.symbol_dim
    input_id += (symbol_id + 1)
    input_id %= (args.symbol_dim + 1)
    net.step_while(args.dt, args.washout_period, u_in=w_in[input_id])

    pert = np.random.randn(*net.x.shape) * 1e-3
    pert[:, :net.dim1] = 0.0
    x_init = net.x.repeat(2, axis=0)
    x_init[1::2] += pert
    net.reset(x_init)

    rec_t, rec_net = simulate(
        net, [0, args.eval_period], dt=args.dt,
        f_in=lambda _t: w_in[symbol_id], prefix="eval_network ")
    d_init = x_init[::2] - x_init[1::2]
    d_eval = rec_net[:, ::2] - rec_net[:, 1::2]
    d_pre = np.linalg.norm(d_init, axis=1)
    d_post = np.linalg.norm(d_eval, axis=2)
    d_ratio = np.log(d_post / d_pre)
    t_over = np.where(np.mean(d_ratio, axis=1) > 0)[0]
    if len(t_over) > 0:
        t_over = t_over.min() * args.dt
    else:
        t_over = np.inf
    return rec_t, rec_net, d_ratio, t_over


if __name__ == '__main__':
    # save network & params
    save_dir = f"{args.output_dir}/init"
    os.makedirs(save_dir, exist_ok=True)
    with open("{}/params.json".format(save_dir), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    while True:
        w_in, net = create_network()
        fig = Figure()
        fig[0].plot_matrix(net.w_net, vmin=-1, vmax=+1, cmap="bwr")
        fig.savefig("{}/matrix.png".format(save_dir))
        # Figure.show()
        fig.close()

        graph = Plotter()
        plot_range = list(itertools.chain(
            range(0, 1), range(net.dim1, net.dim1 + 4)))
        plot_range = pd.unique([_ for _ in plot_range if _ < net.dim])
        is_chaotic = True
        for _id in range(args.symbol_dim):
            rec_t, rec_net, d_ratio, t_over = eval_network(w_in, net, _id)
            print("symbol {}: t_over={:.1f}".format(_id, t_over))

            fig = Figure()
            fig[0].fill_std(
                rec_t, np.mean(d_ratio, axis=1), np.std(d_ratio, axis=1))
            fig[0].set_title("t_over={:.1f}".format(t_over))
            fig[0].line_x(t_over, ls=":", color="k")
            fig[0].line_y(0.0, ls=":", color="k")
            fig.savefig("{}/test/std_{:02d}.png".format(save_dir, _id))

            graph.trj_init(plot_range=plot_range)
            graph.trj_plot(rec_t, rec_net[:, ::2], color="red", lw=0.2)
            graph.trj_plot(rec_t, rec_net[:, 1::2], color="blue", lw=0.2)
            graph.save("trj", "{}/test/eval_{:02d}".format(save_dir, _id))
            fig.close()
            graph.close()
            if not (args.threshold[0] <= t_over <= args.threshold[1]):
                print("too weak chaoticity! t_over > {:.1f}".format(
                    args.threshold[1]))
                print("create network again!")
                is_chaotic = False
                break
        if is_chaotic:
            break
    np.save("{}/w_in.npy".format(save_dir), w_in)
    net.to_pickle("{}/net_init.pkl".format(save_dir))
