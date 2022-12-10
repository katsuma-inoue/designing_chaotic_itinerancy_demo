#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "lorenz", "chongxin", "rossler", "rossler_hyperchaos", "duffing",
    "runge_kutta", "logistic", "bernoulli", "tent", "henon", "reduce_list",
    "GCM", "CML"
]

import numpy as np
import numpy.random as rnd


def lorenz(x, t, a=10.0, b=28.0, c=8.0 / 3.0):
    '''
    Lorenz system (1963)
    '''
    x_dot = np.array(x)
    x_dot[0] = a * (x[1] - x[0])
    x_dot[1] = x[0] * (b - x[2]) - x[1]
    x_dot[2] = x[0] * x[1] - c * x[2]
    return x_dot


def chongxin(t, x, a=10, b=40, k=1, c=2.5, h=4):
    '''
    Chongxin system (2005)
    '''
    x_dot = np.array(x)
    x_dot[0] = a * (x[1] - x[0])
    x_dot[1] = b * x[0] - k * x[0] * x[2]
    x_dot[2] = -c * x[2] + h * x[0] * x[0]


def rossler(t, x, a=0.2, b=0.2, c=5.7):
    '''
    Rossler system (1976)
    '''
    x_dot = np.array(x)
    x_dot[0] = -x[1] - x[2]
    x_dot[1] = x[0] + a * x[1]
    x_dot[2] = b + x[2] * (x[0] - c)
    return x_dot


def rossler_hyperchaos(x, t, a=0.25, b=3.0, c=0.5, d=0.05):
    '''
    Rossler hyperchaos system (1979)
    '''
    x_dot = np.array(x)
    x_dot[0] = -x[1] - x[2]
    x_dot[1] = x[0] + a * x[1] + x[3]
    x_dot[2] = b + x[2] * x[0]
    x_dot[3] = -c * x[2] + d * x[3]
    return x_dot


def duffing(x, t, a=0.25, b=0.3, c=1.0):
    '''
    Duffing equation
    '''
    x_dot = np.array(x)
    x_dot[0] = x[1]
    x_dot[1] = x[0] - x[0]**3 - a * x[1] + b * np.cos(c * t)
    return x_dot


def runge_kutta(func, x0, dt, T, **params):
    '''
    classical Runge-Kutta Method (a.k.a. RK4)

    Args:
        func (function): system equation
        (dx/dt = func(t, x, **params))
        x0 (np.ndarray or list): initial state
        dt (float): time width
        T (float): [description]
        **params :keyward argments for func

    Returns:
        np.ndarray: time list
        np.ndarray: dynamics
    '''
    x = np.array(x0)
    ts = np.arange(0, T, dt)
    record = np.zeros((len(ts), *x.shape))
    for _i, _t in enumerate(ts):
        record[_i] = x
        k1 = dt * func(x, _t, **params)
        k2 = dt * func(x + 0.5 * k1, _t + 0.5 * dt, **params)
        k3 = dt * func(x + 0.5 * k2, _t + 0.5 * dt, **params)
        k4 = dt * func(x + k3, _t + dt, **params)
        x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return ts, record


def logistic(x, a=3.6):
    '''
    Logistic map (1976)
    '''
    return a * x * (1 - x)


def bernoulli(x, n=2):
    '''
    Bernoulli map
    '''
    return (n * x) % 1


def tent(x, mu=2.0):
    '''
    Tent map
    (it is identical to bit shift map when mu=2.0)
    '''
    return mu * min(x, 1 - x)


def henon(x, a=1.4, b=0.3):
    '''
    Henon systems (1976)
    '''
    x_dot = np.array(x)
    x_dot[0] = 1.0 - a * (x[0]**2) + x[1]
    x_dot[1] = b * x[0]
    return x_dot


def reduce_list(func, x0, N, **params):
    '''
    Calculating dynamics of discrete-time dynamical systems

    Args:
        func (function): function (x(t+1) = f(x, **params))
        x0 (list or np.ndarray): initial state
        N (int): time steps
        **params: keyword argments for func
    Returns:
        np.ndarray: dynamics
    '''
    x = np.array(x0)
    record = np.zeros((N, *x.shape))
    for _i in range(N):
        record[_i] = x
        x = func(x, **params)
    return record


class GCM(object):
    '''
    Globally coupled map

    Attributes:
        dim (int): number of nodes
        alpha (float): strength of nodes
        epsilon (float): strength of connection
        x_init (np.ndarray): initial state
    '''
    def __init__(self, dim, alpha, epsilon,
                 x_init=None, activation=None):
        self.dim = dim
        self.alpha = alpha
        self.epsilon = epsilon
        if x_init is None:
            self.x_init = rnd.uniform(-1, 1, size=(self.dim,))
        else:
            self.x_init = x_init
        self.x = self.x_init
        if activation is None:
            self.f = lambda x, a=self.alpha: -a * x**2 + 1
        else:
            self.f = activation

    def step(self, u_in=None):
        if u_in is not None:
            self.x += u_in
        self.y = self.f(self.x)
        self.x = (1 - self.epsilon) * self.y + self.epsilon * np.mean(self.y)

    def step_while(self, num_step, init_step=0):
        for _ in range(init_step, num_step):
            self.step()


class CML(object):
    '''
    Coupled map lattice

    Attributes:
        dim (int): number of nodes
        alpha (float): strength of nodes
        epsilon (float): strength of connection
        x_init (np.ndarray): initial state
    '''
    def __init__(self, dim, alpha, epsilon,
                 x_init=None, activation=None, is_ring=True):
        self.dim = dim
        self.alpha = alpha
        self.epsilon = epsilon
        if x_init is None:
            self.x_init = rnd.uniform(-1, 1, size=(self.dim,))
        else:
            self.x_init = x_init
        self.x = self.x_init
        if activation is None:
            self.f = lambda x, a=self.alpha: -a * x**2 + 1
        else:
            self.f = activation
        self.w_net = (1 - self.epsilon) * np.eye(self.dim)
        self.w_net += np.roll(self.epsilon / 2 * np.eye(self.dim), 1, axis=0)
        self.w_net += np.roll(self.epsilon / 2 * np.eye(self.dim), -1, axis=0)
        if not is_ring:
            self.w_net[0, self.dim - 1] = 0
            self.w_net[self.dim - 1, 0] = 0

    def step(self, u_in=None):
        if u_in is not None:
            self.x += u_in
        self.y = self.f(self.x)
        self.x = self.w_net.dot(self.y)

    def step_while(self, num_step, init_step=0):
        for _ in range(init_step, num_step):
            self.step()
