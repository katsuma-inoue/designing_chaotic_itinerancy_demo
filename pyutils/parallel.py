#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["print", "Worker", "for_each", "multi_process", "multi_thread"]

import os
import time
from pathos.helpers import mp
from pathos.pools import ProcessPool, ThreadPool, SerialPool, ParallelPool
from multiprocessing import Manager, Pipe, Pool, Process, Queue, Value

# override builtin functions
import builtins as __builtin__
begin = time.time()


def _print(*args, **kwargs):
    __builtin__.print(
        "\x1b[2K[{:8.3f}]".format(time.time() - begin), *args, **kwargs)


print = _print


class Worker(object):
    def __init__(self, _obj, verbose=False):
        self.pipe_func_parent, self.pipe_func_child = Pipe(True)
        self.pipe_attr_parent, self.pipe_attr_child = Pipe(False)
        self.obj = _obj
        self.obj.__class__.__copy__ = lambda _self: _self
        self.verbose = verbose
        p = Process(target=self._process, args=(
            self.pipe_func_child, self.pipe_attr_child))
        p.daemon = True
        p.start()

    def _process(self, _pipe_func, _pipe_attr):
        while True:
            func_name, store_value, args, kwargs = _pipe_func.recv()
            if self.verbose:
                print("({:5d}) call {}({},{})".format(
                    os.getpid(), func_name, args, kwargs))
            result = getattr(self.obj, func_name)(*args, **kwargs)
            if (func_name == "__copy__") or (func_name == "__getattribute__"):
                _pipe_attr.send(result)
            elif store_value:
                _pipe_func.send(result)

    def call(self, func_name, *args, store_value=True, **kwargs):
        self.pipe_func_parent.send((func_name, store_value, args, kwargs))

    def put(self, *args, **kwargs):
        self.call("__call__", args, kwargs)

    def get(self):
        return self.pipe_func_parent.recv()

    def getattr(self, attr_name):
        self.call("__getattribute__", attr_name)
        return self.pipe_attr_parent.recv()

    def getcopy(self):
        self.call("__copy__")
        return self.pipe_attr_parent.recv()


'''
based on pathos multiprocessing / thread functions

*** example1 ***
    f = lambda t, r=1000000 : sum(range(t, t + r))
    multi_process(f, [[_] for _ in range(1000), nodes])

*** example2 ***
    def generator():
       for _i in range(1000):
          yield (_i,)

    multi_process(f, generator(), nodes)
'''


def for_each(_func, args_list, expand=False, verbose=True):
    # wrapping functions
    def _func_wrapper(task_id, args):
        if verbose:
            print("\x1b[32mNo.{}'s task begin\x1b[0m".format(task_id + 1))
        if expand:
            return _func(*args)
        else:
            return _func(args)
    # main processes
    return [_func_wrapper(_id, args) for _id, args in enumerate(args_list)]


def multi_process(_func, args_list, nodes=None, expand=False,
                  verbose=True, append_id=False):
    # wrapping functions
    def _func_wrapper(args):
        cp = mp.current_process()
        if verbose:
            print("\x1b[32mNo.{}'s task begin\x1b[0m".format(args[0] + 1))
        if expand:
            if append_id:
                return _func(*args[1], worker_id=cp._identity)
            else:
                return _func(*args[1])
        else:
            if append_id:
                return _func(args[1], worker_id=cp._identity)
            else:
                return _func(args[1])
    # main processes
    with ProcessPool(nodes=nodes) as p:
        try:
            res = p.amap(_func_wrapper, enumerate(args_list))
            return res.get()
        except Exception as err_msg:
            print(err_msg)
            p.terminate()
        except KeyboardInterrupt:
            p.terminate()


def multi_thread(_func, args_list, nodes=None, expand=False, verbose=True):
    # wrapping functions
    def _func_wrapper(args):
        if verbose:
            print("\x1b[32mNo.{}'s task begin\x1b[0m" % (args[0] + 1))
        if expand:
            return _func(*args[1])
        else:
            return _func(args[1])
    # main processes
    with ThreadPool(nodes=nodes) as p:
        try:
            res = p.amap(_func_wrapper, enumerate(args_list))
            return res.get()
        except Exception as err_msg:
            print(err_msg)
            p.terminate()
        except KeyboardInterrupt:
            p.terminate()
