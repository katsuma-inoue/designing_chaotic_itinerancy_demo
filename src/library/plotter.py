#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["Plotter"]

import os
import itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pyutils.figure import Figure

import matplotlib
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
sns.set(font_scale=1.5, font="Arial")
sns.set_palette("tab10")
sns.set_style("whitegrid", {'grid.linestyle': '--'})


class Plotter(object):
    def __init__(self):
        self.fig = {}

    def show(self):
        Figure.show()

    def save(self, fig_name, save_name, extensions=["png"]):
        for _ext in extensions:
            transparent = (_ext == "pdf")
            if type(self.fig[fig_name]) is Figure:
                self.fig[fig_name].savefig(
                    "{}.{}".format(save_name, _ext), transparent=transparent)
            else:
                self.fig[fig_name].figure.savefig(
                    "{}.{}".format(save_name, _ext), transparent=transparent)

    def close(self):
        for _key, _val in self.fig.items():
            _val.close()
        self.fig = {}

    def trj_init(self, plot_range=None, plot_num=5):
        if plot_range is None:
            self.trj_range = list(range(plot_num))
        else:
            self.trj_range = list(plot_range)
        plot_num = len(self.trj_range)
        self.fig["trj"] = Figure(figsize=(10, plot_num * 1.5))
        self.fig["trj"].create_grid((plot_num, 1), wspace=0.0, hspace=0.0)
        for _i in range(plot_num):
            # self.fig["trj"][_i].set_ylim([-1.1, 1.1])
            # self.fig["trj"][_i].set_yticks([-1.0, 0.0, 1.0])
            self.fig["trj"][_i].set_yticklabels([])
            if _i < plot_num - 1:
                self.fig["trj"][_i].set_xticklabels([])
            self.fig["trj"][_i].grid(True)

    def trj_plot(self, rec_t, rec_net, t_range=slice(None), **kwargs):
        for _x in np.rollaxis(rec_net, 1):
            for _i, _id in enumerate(self.trj_range):
                # print(rec_t[t_range].shape)
                self.fig["trj"][_i].plot(
                    rec_t[t_range], _x[t_range, _id], **kwargs)
                self.fig["trj"][_i].set_xlim([rec_t[0], rec_t[-1]])

    def trj_fill(self, t_from, t_until, **kwargs):
        for _i, _id in enumerate(self.trj_range):
            self.fig["trj"][_i].fill_x(t_from, t_until, **kwargs)

    def feed_init(self, model):
        self.feed_model = model
        self.feed_dim = model.classes_.size
        self.fig["feed"] = Figure(figsize=(10, 10))
        self.fig["feed"].create_grid(
            (3, 1), wspace=0.0, hspace=0.1,
            height_ratios=(1, 1, 1))
        for _i in range(2):
            self.fig["feed"][_i].create_grid(
                (self.feed_dim, 1), wspace=0.0, hspace=0.0)

    def feed_plot(self, rec_t, rec_net, **kwargs):
        tab10 = matplotlib.pyplot.get_cmap("tab10")
        out_label = []
        for _x in np.rollaxis(rec_net, 1):
            out_raw = self.feed_model.decision_function(_x)
            if self.feed_dim == 2:
                out_raw = np.tile(out_raw, (2, 1)).T
            out_prob = self.feed_model.predict_proba(_x)
            out_label.append(self.feed_model.predict(_x))
            for _i in range(self.feed_dim):
                self.fig["feed"][0][_i].plot(rec_t, out_raw[:, _i], **kwargs)
                self.fig["feed"][1][_i].plot(rec_t, out_prob[:, _i], **kwargs)
                self.fig["feed"][0][_i].set_xlim([rec_t[0], rec_t[-1]])
                self.fig["feed"][1][_i].set_xlim([rec_t[0], rec_t[-1]])
                self.fig["feed"][0][_i].set_xticklabels([])
                self.fig["feed"][1][_i].set_xticklabels([])
                self.fig["feed"][1][_i].set_yticklabels([])
        out_label = np.array(out_label).astype(float)
        out_label[out_label < 0] = np.nan
        self.fig["feed"][2].plot_matrix(
            out_label, x=rec_t, aspect="auto",
            cmap="tab10", zscale="discrete", norm=np.arange(11) - 0.5,
            colorbar=False)
        self.fig["feed"][2].set_yticks([])

    def symbol_init(self, model):
        self.symbol_model = model
        self.symbol_dim = model.classes_.size
        self.fig["symbol"] = Figure(figsize=(16, 8))
        self.fig["symbol"].create_grid((1, 2), wspace=0.1, hspace=0.0)

    def symbol_plot(self, rec_t, rec_net):
        rec_period = []
        df_transition = pd.DataFrame(
            columns=self.symbol_model.classes_,
            index=self.symbol_model.classes_).fillna(0).astype(float)
        for _x in np.rollaxis(rec_net, 1):
            label_list = self.symbol_model.predict(_x)
            label_group = [list(_group)
                           for _key, _group in itertools.groupby(label_list)]
            for _group in label_group[1:-1]:
                rec_period.append([_group[0], len(_group)])
            for _pre, _post in zip(label_group[1:-2], label_group[2:-1]):
                df_transition[_post[0]][_pre[0]] += 1
        df_transition = df_transition.apply(lambda x: x / x.sum(), axis=1)
        df_period = pd.DataFrame(rec_period,
                                 columns=["symbol", "period"])
        sns.boxplot(x="symbol", y="period", data=df_period,
                    ax=self.fig["symbol"][0], color="cyan")
        sns.pointplot(x="symbol", y="period", data=df_period,
                      ax=self.fig["symbol"][0])
        self.fig["symbol"][0].set_ylim([0, None])
        df_g = df_period.groupby("symbol")
        print(df_g.agg(['mean', 'std']))
        sns.heatmap(df_transition, annot=True, fmt=".2f",
                    ax=self.fig["symbol"][1], vmin=0.0, vmax=1.0)
        return df_transition, df_period

    def pca_init(self, model):
        fig = Figure(figsize=(8, 8))
        self.fig["pca"] = fig[0].convert_3d()
        self.pca_model = model
        self.pca_dim = model.classes_.size

    def pca_plot(self, rec_t, rec_net, pca_step=27, pca_width=100):
        pca_data, color_data = [], []
        tab10 = matplotlib.pyplot.get_cmap("tab10")
        for _x in np.rollaxis(rec_net, 1):
            pca, color = [], []
            label_list = self.pca_model.predict(_x)
            for _i in range(int((len(_x) - pca_width) / pca_step)):
                pca.append(
                    _x[_i * pca_step:_i * pca_step + pca_width].flatten())
                color.append(tab10(label_list[_i * pca_step] % 10))
                print("pca {} steps".format(_i), end="\r")
            pca_data.append(np.array(pca))
            color_data.append(color)

        pca = PCA(n_components=3)
        pca.fit(np.concatenate(pca_data, axis=0))
        for _pca, _color in zip(pca_data, color_data):
            _pos = pca.transform(_pca)
            self.fig["pca"].scatter(
                _pos[:, 0], _pos[:, 1], _pos[:, 2], color=_color)
            self.fig["pca"].plot(
                _pos[:, 0], _pos[:, 1], _pos[:, 2], lw=0.2)
