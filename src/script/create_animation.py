#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import joblib
import string
import inquirer
import argparse
import itertools
import numpy as np

sys.path.append(".")

from src.library.plotter import Plotter

from pyutils.tqdm import tqdm, trange
from pyutils.figure import Figure

import matplotlib.pyplot as plt
import matplotlib.animation as animation

parser = argparse.ArgumentParser()
parser.add_argument("load_dir", type=str)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument('--func_name', type=str, nargs="+",
                    default=["out_2d", "symbol", "net"])
parser.add_argument("--func_period", type=int, nargs="+", default=None)
parser.add_argument('--time_range', type=int, nargs="+", default=None)
parser.add_argument('--sample_range', type=int, nargs="+", default=None)
parser.add_argument('--figsize', type=float, nargs="+", default=None)
parser.add_argument('--xlim', type=float, nargs="+", default=None)
parser.add_argument('--ylim', type=float, nargs="+", default=None)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--t_width', type=float, default=2000.0)
parser.add_argument('--skip', type=int, default=30)
parser.add_argument('--dpi', type=int, default=120)
parser.add_argument('--per_step', type=float, default=1.0)
parser.add_argument('--off_color', action='store_true')
args = parser.parse_args()


def animate_out_2d(
        rec_t, rec_net, rec_out, rec_symbol, time_list, sample_list,
        figsize=None, xlim=None, ylim=None, off_color=False,
        skip=30, t_width=2000.0, dt=1.0, per_step=1.0, **kwargs):
    if figsize is None:
        figsize = [8, 8]
    if xlim is None:
        xlim = [-4, 4]
    if ylim is None:
        ylim = [-4, 4]
    fig = Figure(figsize=figsize)
    fig[0].set_xlim(xlim)
    fig[0].set_ylim(ylim)
    fig[0].set_xticks([-4, -2, 0, 2, 4])
    fig[0].set_yticks([-4, -2, 0, 2, 4])
    print(rec_out.shape)

    marker_list = ['o']
    # marker_list = ['o', 'v', '^', '<', '>']
    im_list = [fig[0].scatter([], [], s=2.0) for _ in sample_list]
    pt_list = [fig[0].plot([], [], marker=marker_list[_ % len(marker_list)],
                           color="black", markersize=5, ls="")[0]
               for _ in sample_list]
    time_text = fig[0].text(0.05, 0.95, "", transform=fig[0].transAxes)
    input_text = fig[0].text(0.05, 0.90, "", transform=fig[0].transAxes)
    color_list = np.array(
        [plt.get_cmap("tab10")(_ % 10) for _ in range(10)] + ["gray"])
    label_list = np.array([_ for _ in string.ascii_uppercase + "-"])

    frame_num = int(time_list.shape[0] / skip) - 1
    pbar = trange(frame_num, desc="animate_out_2d ")

    def _draw(_n):
        n_now = time_list[(_n + 1) * skip]
        n_pre = max(0, n_now - int(t_width / dt))
        pbar.update(1)
        for _im, _pt, _id in zip(im_list, pt_list, sample_list):
            _im.set_offsets(rec_out[n_pre:n_now, _id])
            if not off_color:
                _im.set_facecolors(color_list[rec_symbol[n_pre:n_now, _id]])
            _pt.set_data(rec_out[n_now, _id])
        label = "".join(label_list[rec_symbol[n_now][sample_list]])
        time_text.set_text("t = {:.0f}".format(rec_t[n_now]))
        input_text.set_text("symbol: {}".format(label))
    ani = animation.FuncAnimation(
        fig._fig, _draw, frames=frame_num, interval=per_step * skip)
    fig._fig.subplots_adjust(0, 0, 1, 1)
    # Figure.show(tight_layout=False)
    return ani


def animate_net(
        rec_t, rec_net, rec_out, rec_symbol, time_list, sample_list,
        figsize=None, xlim=None, ylim=None, off_color=False,
        skip=10, t_width=2000.0, dt=1.0, per_step=1.0, **kwargs):
    if figsize is None:
        figsize = [8, 10]
    fig = Figure(figsize=figsize)

    input_dim = 2
    # innate_dim = 8
    # plot_range = list(range(input_dim))
    # plot_range += list(range(500, 500 + innate_dim))
    plot_range = list(range(10))
    fig.create_grid((len(plot_range), 1), hspace=0.0)

    im_list = []
    for _id in sample_list:
        im_list.append([
            fig[_].plot([], [])[0] for _ in range(len(plot_range))])

    for _i, _node in enumerate(plot_range):
        if _i < len(plot_range) - 1:
            fig[_i].set_xticklabels([])
        else:
            fig[_i].set_xlabel("t")
        if _i < input_dim:
            val_max = rec_net[:, :, _i].max()
            val_min = rec_net[:, :, _i].min()
            val = 1.1 * max(abs(val_max), abs(val_min))
            fig[_i].set_ylim([-val, val])
        else:
            fig[_i].set_ylim([-1.1, 1.1])
        # fig[_i].set_ylabel(r"$x_{" + str(_i + 1) + "}$")
        fig[_i].set_yticklabels([])
        fig[_i].grid()

    frame_num = int(time_list.shape[0] / skip) - 1
    pbar = trange(frame_num, desc="animate_net ")

    def _draw(_n):
        n_now = time_list[(_n + 1) * skip]
        n_pre = max(0, n_now - int(t_width / dt))
        pbar.update(1)
        x_range = [rec_t[n_now] - t_width, rec_t[n_now]]
        for _i, _node in enumerate(plot_range):
            fig[_i].set_xlim(x_range)
            for _j, (_im, _id) in enumerate(zip(im_list, sample_list)):
                _im[_i].set_data(
                    rec_t[n_pre:n_now],
                    rec_net[n_pre:n_now, _id, _node])

    ani = animation.FuncAnimation(
        fig._fig, _draw, frames=frame_num, interval=per_step * skip)
    fig._fig.subplots_adjust(0, 0, 1, 1)
    # Figure.show(tight_layout=False)
    return ani


def animate_symbol(
        rec_t, rec_net, rec_out, rec_symbol, time_list, sample_list,
        figsize=None, xlim=None, ylim=None, off_color=False,
        skip=10, t_width=2000.0, dt=1.0, per_step=1.0, **kwargs):
    if figsize is None:
        figsize = [8, 3.0]
    fig = Figure(figsize=figsize)
    tab10 = plt.get_cmap("tab10")
    _symbol = rec_symbol.astype(float)
    _symbol[rec_symbol < 0] = np.nan
    fig[0].plot_matrix(
        _symbol.T[sample_list], x=rec_t, aspect="auto",
        cmap=tab10, zscale="discrete", norm=np.arange(11) - 0.5,
        colorbar=False, extent=[rec_t[0], rec_t[-1], 0, 1])
    fig[0].set_yticklabels([])
    fig[0].grid(False)

    frame_num = int(time_list.shape[0] / skip) - 1
    pbar = trange(frame_num, desc="animate_symbol ")

    def _draw(_n):
        n_now = time_list[(_n + 1) * skip]
        n_pre = max(0, n_now - int(t_width / dt))
        pbar.update(1)
        x_range = [rec_t[n_now] - t_width, rec_t[n_now]]
        fig[0].set_xlim(x_range)
    ani = animation.FuncAnimation(
        fig._fig, _draw, frames=int(time_list.shape[0] / skip) - 1,
        interval=per_step * skip)
    fig._fig.subplots_adjust(0, 0, 1, 1)
    # Figure.show(tight_layout=False)
    return ani


if __name__ == '__main__':
    questions = [
        inquirer.List(
            'record', message="choose record file",
            choices=list(glob.glob(f"{args.load_dir}/record/*/record.pkl"))
        )]
    answers = inquirer.prompt(questions)

    with open(answers["record"], mode="rb") as f:
        rec_t, rec_net, rec_out, rec_symbol = joblib.load(f)

    if args.sample_range is None:
        sample_list = np.arange(rec_net.shape[1])
        suffix = "all"
    else:
        sample_list = slice(*args.sample_range)
        sample_list = np.arange(rec_net.shape[1])[sample_list].astype(int)
        suffix = "-".join(map(str, sample_list))

    if args.time_range is None:
        time_list = slice(rec_t.shape[0])
    else:
        time_list = slice(*args.time_range)
        suffix += "_" + ",".join(map(str, args.time_list))
    time_list = np.arange(rec_t.shape[0])[time_list].astype(int)

    if args.func_period is not None:
        for _i in range(rec_symbol.shape[1]):
            current_id = 0
            new_symbol = np.array(rec_symbol[:, _i])
            for _key, _group in itertools.groupby(new_symbol):
                length = len(list(_group))
                begin_id = current_id + \
                    args.func_period[_key % len(args.func_period)]
                end_id = current_id + length
                new_symbol[begin_id:end_id] = -1
                current_id += length
            rec_symbol[:, _i] = new_symbol

    if args.save_dir is None:
        args.save_dir = os.path.dirname(answers["record"])

    for _func_name in args.func_name:
        ani = eval("animate_{}".format(_func_name))(
            rec_t, rec_net, rec_out, rec_symbol,
            time_list, sample_list, **args.__dict__)
        if ani is not None:
            ani.save("{}/{}_{}.mp4".format(
                args.save_dir, _func_name, suffix),
                writer="ffmpeg", dpi=args.dpi)
