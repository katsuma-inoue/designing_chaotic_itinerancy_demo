#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "count_if", "cross_correlation", "sample_cross_correlation",
    "fft", "rfft", "interpft", "mean_square_error",
    "normalized_mean_square_error", "peak_signal_to_noise_ratio",
    "ridge_regression", "select_best_regularization",
    "optimize_ridge_criteria",
    "curve_fitting", "sin_cos_basis", "concatenate_with_pad",
    "memory_capacity"
]

import cmath
import numpy as np
import scipy.signal
import scipy.optimize
from collections.abc import Iterable
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, KFold

from pyutils.tqdm import tqdm, trange

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')


def count_if(x, f):
    return sum(f(_) for _ in x)


def cross_correlation(s1, s2, mode='valid'):
    return scipy.signal.correlate(s1, s2, mode=mode)


def sample_cross_correlation(x1, x2, mode='valid'):
    num = min(len(x1), len(x2))
    s1 = (np.array(x1) - np.mean(x1))[:num]
    s2 = (np.array(x2) - np.mean(x2))[:num]
    c = scipy.signal.correlate(s1, s2, mode=mode)
    return c * (1 / (num * np.std(s1) * np.std(s2)))


def fft(x, d=1.0, window=None):
    num = len(x)
    if window is not None:
        data = x * scipy.signal.get_window(window, num)
    else:
        data = x
    sp = np.fft.fft(data, norm=None)
    sp /= len(sp)
    freq = np.fft.fftfreq(num, d=d)
    amp = [abs(_) for _ in sp]
    pha = [cmath.phase(_) for _ in sp]
    return freq, amp, pha


def rfft(x, d=1.0, window=None):
    num = len(x)
    if window is not None:
        data = x * scipy.signal.get_window(window, num)
    else:
        data = x
    sp = np.fft.rfft(x, norm=None)
    sp /= len(sp)
    freq = np.fft.rfftfreq(num, d=d)
    amp = [abs(_) for _ in sp]
    pha = [cmath.phase(_) for _ in sp]
    return freq, amp, pha


def interpft(x, d=1.0, window=None):
    num = len(x)
    if window is not None:
        data = x * scipy.signal.get_window(window, num)
    else:
        data = x
    sp = np.fft.fft(x, norm=None)
    freq = np.fft.fftfreq(num, d=d)
    return lambda t, dt=d: np.real(sum([
        _sp * np.exp(1j * 2 * np.pi * _f * t)
        for _i, (_f, _sp) in enumerate(zip(freq, sp))])) / num


def mean_square_error(x, y, axis=1):
    x, y = map(np.array, [x, y])
    if x.ndim > 1:
        err = np.linalg.norm(x - y, axis=axis)
    else:
        err = x - y
    return (err**2).mean()


def root_mean_square_error(x, y, axis):
    return np.sqrt(mean_square_error(x, y, axis))


def normalized_mean_square_error(x, y, axis=None):
    mse = mean_square_error(x, y, axis)
    var = np.mean(x, axis=axis) * np.mean(y, axis=axis)
    return mse / var


def peak_signal_to_noise_ratio(x, y, axis=None):
    x, y = map(np.array, [x, y])
    ms = mean_square_error(x, y, axis)
    ma = x.max(axis=axis)
    return 10 * np.log10(ma**2 / ms)


def ridge_regression(X, Y, alpha=1e-8):
    dim_data, dim_in = X.shape
    A = X.T.dot(X) + alpha * np.eye(dim_in)
    B = X.T.dot(Y)
    return np.linalg.solve(A, B)


def select_best_regularization(
        X, Y, n_splits=5, resolution=None, x0=None, verbose=True):
    X_data = X - np.mean(X, axis=0)
    Y_data = Y - np.mean(Y, axis=0)
    rank = np.linalg.matrix_rank(X_data)
    C_data = X.T.dot(X_data)
    I_mat = np.eye(X_data.shape[1])
    print("rank: {}".format(rank))
    eig_vals = scipy.linalg.eigvalsh(C_data)

    def dof_func(alpha):
        return sum([_e / (_e + alpha) for _e in eig_vals])

    if resolution is None:
        resolution = rank

    if x0 is None:
        x0 = np.mean(eig_vals[eig_vals > 0.0][:2])

    dof_list = np.linspace(1, rank, resolution)
    alpha_list, error_list = np.zeros(resolution), np.zeros(resolution)

    pbar = tqdm(enumerate(dof_list), total=resolution)
    for _i, _dof in pbar:
        def _func(_x):
            if 0 <= _x:
                return dof_func(_x) - _dof
            else:
                return dof_func(0.0)
        kf = KFold(n_splits=n_splits)
        alpha = scipy.optimize.fsolve(_func, x0)[0]
        validation_errors = []
        for train_idx, eval_idx in kf.split(X_data):
            X_train, X_eval = X_data[train_idx], X_data[eval_idx]
            Y_train, Y_eval = Y_data[train_idx], Y_data[eval_idx]
            # A = X_train.T.dot(X_train) + alpha * I_mat
            # B = X_train.T.dot(Y_train)
            weight = np.linalg.lstsq(
                X_train + alpha * np.random.randn(X_train.shape), Y_train)[0]
            Y_out = X_eval.dot(weight)
            mse = ((Y_out - Y_eval) ** 2).mean()
            validation_errors.append(mse)
        error = np.mean(validation_errors)
        pbar.set_description("α: {:.2e}, MSE: {:.2e}".format(alpha, error))
        alpha_list[_i] = alpha
        error_list[_i] = error
    pbar.close()
    positive = alpha_list > 0
    id_min = np.argmin(error_list[positive])
    dof_min = dof_list[positive][id_min]
    alpha_min = alpha_list[positive][id_min]
    error_min = error_list[positive][id_min]

    print("α_min: {:.2e} (dof: {:.2e}, MSE: {:.2e})".format(
        alpha_min, dof_min, error_min))
    if verbose:
        return alpha_min, (alpha_list, error_list)
    else:
        return alpha_min


class Criteria(object):
    def __init__(self, X, Y):
        self.X, self.Y = np.array(X), np.array(Y)
        self.X -= np.average(self.X, axis=0)
        self.Y -= np.average(self.Y, axis=0)
        self.dim_data, self.dim_param = self.X.shape
        print("input_shape:{}, target_shape:{}".format(
            self.X.shape, self.Y.shape))
        # self.X_train, self.X_eval, self.Y_train, self.Y_eval = \
        #     train_test_split(
        #         self.X, self.Y, test_size=0.33,
        #         shuffle=True, random_state=None)
        self.rank = np.linalg.matrix_rank(self.X)
        print("rank:{}".format(self.rank))
        self.C_mat = self.X.T.dot(self.X)
        self.S_mat = self.X.T.dot(self.Y)
        self.I_mat = np.eye(self.dim_param)
        self.eig_vals = scipy.linalg.eigvalsh(self.C_mat)
        self.eig_vals[self.eig_vals < 0] = 0.0

    def _ridge(self, alpha):
        solution, res, _rank, _s = np.linalg.lstsq(
            self.C_mat + alpha * self.I_mat, self.S_mat)
        return solution, res

    def AIC(self, alpha):
        df = sum([_e / (_e + alpha) for _e in self.eig_vals])
        weight, rss = self._ridge(alpha)
        aic = self.dim_data * np.log(rss.sum()) + 2 * df
        return aic

    def BIC(self, alpha):
        df = sum([_e / (_e + alpha) for _e in self.eig_vals])
        weight, rss = self._ridge(alpha)
        bic = self.dim_data * np.log(rss.sum()) + 2 * df
        return bic


def optimize_ridge_criteria(
        X, Y, num_split=None, criteria="AIC", x0=None, verbose=False):
    info_cret = Criteria(X, Y)
    dim_param = X.shape[1]
    num_split = info_cret.rank if num_split is None else num_split

    def dof_func(alpha):
        return sum([_e / (_e + alpha) for _e in info_cret.eig_vals])
    dof_list = np.linspace(1, info_cret.rank, num_split)
    cret_list, alpha_list = [], []
    pbar = tqdm(dof_list)
    if x0 is None:
        x0 = np.mean(info_cret.eig_vals[info_cret.eig_vals > 0.0][:2])

    for _dof in dof_list:
        def _func(_x):
            if 0 <= _x:
                return dof_func(_x) - _dof
            else:
                return dof_func(0.0)
        _alpha = scipy.optimize.fsolve(_func, x0)
        _cret = getattr(info_cret, criteria)(_alpha)
        pbar.set_description("alpha:{:.3e}".format(_alpha[0]))
        pbar.update(1)
        alpha_list.append(_alpha[0])
        cret_list.append(_cret[0])
    pbar.close()
    cret_list = np.array(cret_list)
    alpha_list = np.array(alpha_list)

    positive = alpha_list > 0
    id_min = np.argmin(cret_list[positive])
    alpha_min = alpha_list[positive][id_min]
    dof_min = dof_list[np.argmin(cret_list[positive])]

    print("\nalpha_min:{:2.3e}, dof_min:{}".format(alpha_min, dof_min))
    if verbose:
        return alpha_min, (alpha_list, cret_list)
    else:
        return alpha_min


def curve_fitting(func_list, x, y, alpha=0.0, return_weight=False):
    x_proj = np.array([_f(x) for _f in func_list]).T
    w = ridge_regression(x_proj, y, alpha)

    def f(t):
        return np.array([_f(t) for _f in func_list])
    if return_weight:
        return w
    else:
        return lambda t: w.dot(f(t))


def sin_cos_basis(dim, k=1.0):
    func_list = [lambda t, f=k * _f: np.sin(2 * np.pi * f * t)
                 for _f in range(1, dim + 1)]
    func_list += [lambda t, f=k * _f: np.cos(2 * np.pi * f * t)
                  for _f in range(0, dim + 1)]
    return func_list


def concatenate_with_pad(args, axis=0):
    assert all([_.ndim == 2 for _ in args]
               ), "dimensions of all elements must be 2."
    max_dim = max([_.shape[(axis + 1) % 2] for _ in args])

    def _pad_width(_data):
        if axis == 0:
            return [(0, 0), (0, max_dim - _data.shape[1])]
        else:
            return [(0, max_dim - _data.shape[0]), (0, 0)]
    data_list = [np.pad(_, _pad_width(_), "constant") for _ in args]
    return np.concatenate(data_list, axis=axis)


def memory_capacity(input_list, output_list, K=None, alpha=1e-2):
    '''
    input_list (np.ndarray): list of input (1d-array)
    output_list (np.ndarray): list of internal state (1d-array)
    K (int): the maximum past time step
    '''
    assert len(input_list) == len(output_list), "lengths must be same."
    num_data = len(output_list)
    if K is None:
        K = int(num_data * 0.1)
    train_length = (num_data - K) // 2
    X = np.c_[output_list, input_list]
    num_param = X.shape[1]
    X_train = X[K:K + train_length]
    X_test = X[K + train_length:]
    H = np.linalg.inv(
        X_train.T.dot(X_train) + alpha * np.eye(num_param)).dot(X_train.T)
    result = []
    for _i in range(K):
        if _i is 0:
            _y_train = input_list[K:K + train_length]
            _y_test = input_list[K + train_length:]
        else:
            _y_train = input_list[K - _i:K + train_length - _i]
            _y_test = input_list[K + train_length - _i:-_i]
        _y_pred = X_test.dot(H.dot(_y_train))
        cs = sample_cross_correlation(_y_pred, _y_test)[0]
        result.append(cs)
    return result
