#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import joblib
import inquirer
import datetime
import argparse
import numpy as np

sys.path.append(".")

from pyutils.figure import Figure
from src.library.plotter import Plotter
from src.library.simulate import simulate

import seaborn as sns
import matplotlib.colors as clr
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("load_dir", type=str)
parser.add_argument("--w_in_path", type=str, default="w_in.npy")
parser.add_argument("--net_path", type=str, default="net_term.pkl")
parser.add_argument("--symbol_list", type=int, nargs="+", default=None)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--washout_period", type=float, default=1000.0)
parser.add_argument("--switch_period", type=int, nargs="+", default=[])
parser.add_argument("--eval_num", type=int, default=10)
parser.add_argument("--eval_period", type=float, default=50000.0)
parser.add_argument("--eval_pert", type=float, default=1e-6)
args = parser.parse_args()


def eval_readout(w_in, net, w_feed, w_out):
    if args.eval_pert is None:
        net.reset(np.random.randn(args.eval_num, net.dim))
        net.step_while(args.dt, args.washout_period)
    else:
        net.reset(np.random.randn(1, net.dim))
        net.step_while(args.dt, args.washout_period)
        pert = args.eval_pert * np.random.randn(args.eval_num, net.dim)
        pert[:, :net.dim1] = 0.0
        net.reset(pert + net.x)

    def f_feed(_t, _x):
        return w_in[w_feed.predict(_x)]

    record_range = [0, args.eval_period]
    rec_t, rec_net = simulate(net, record_range, dt=args.dt, f_feed=f_feed)
    rec_out = np.array([w_out.predict(_x) for _x in rec_net])
    rec_symbol = np.array([w_feed.predict(_x) for _x in rec_net])
    return rec_t, rec_net, rec_out, rec_symbol


if __name__ == '__main__':
    # loading data
    w_in = np.load("{}/{}".format(args.load_dir, args.w_in_path))
    with open("{}/{}".format(
            args.load_dir, args.net_path), mode="rb") as f:
        net = joblib.load(f)
    questions = [
        inquirer.List(
            'feedback', message="choose feedback file",
            choices=list(glob.glob(f"{args.load_dir}/*/w_feed.pkl"))
        ),
        inquirer.List(
            'readout', message="choose readout file",
            choices=list(glob.glob(f"{args.load_dir}/*/w_out.pkl"))
        )]
    answers = inquirer.prompt(questions)
    with open(answers["feedback"], mode="rb") as f:
        w_feed = joblib.load(f)
    with open(answers["readout"], mode="rb") as f:
        w_out = joblib.load(f)

    save_dir = "/record/{0:%Y-%m-%d_%H:%M:%S}".format(datetime.datetime.now())
    save_dir = f"{args.load_dir}/{save_dir}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"save to {save_dir}")

    rec_t, rec_net, rec_out, rec_symbol = eval_readout(
        w_in, net, w_feed, w_out)
    selected_id = list(range(0, 2)) + list(range(net.dim1, net.dim1 + 8))
    rec_net = rec_net[:, :, selected_id]

    fig = Figure(figsize=(10, 4))
    fig[0].plot_matrix(
        rec_symbol.T, x=rec_t, aspect="auto",
        cmap="tab10", zscale="discrete",
        norm=np.arange(11) - 0.5,
        colorbar=False)
    fig.savefig("{}/symbol_dynamics.png".format(save_dir))

    with open("{}/params.json".format(save_dir), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    with open("{}/record.pkl".format(save_dir), mode="wb") as f:
        joblib.dump([rec_t, rec_net, rec_out, rec_symbol], f, compress=True)
