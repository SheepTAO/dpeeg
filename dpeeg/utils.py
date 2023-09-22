#!/usr/bin/env python
# coding: utf-8

"""
    Util functions for dpeeg.
    
    @Author  : SheepTAO
    @Time    : 2023-09-15
"""


import os
import mne
import torch
import random
import functools
import numpy as np
from .tools.logger import Logger
from typing import (
    Any,
    Union,
    Callable,
    TypeVar,
)


# Life, the Universe, and Everything
DPEEG_SEED : int = 42
DPEEG_LOGGING_LEVEL = 'INFO'
DPEEG_DIR = os.path.join(os.path.expanduser('~'), 'dpeeg')


loger = Logger('dpeeg', flevel=None)


# Provide help for static type checkers:
# https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
_FuncT = TypeVar('_FuncT', bound=Callable[..., Any])


def verbose(func : _FuncT) -> _FuncT:
    '''Verbose decorator to allow functions to override log-level.

    Parameters
    ----------
    func : callable
        Function to be decorated by setting the verbosity level.

    Returns
    -------
    dec : callable
        The decorated function.
    '''
    @functools.wraps(func)
    def inner(*args, **kwargs) -> _FuncT:
        kwargs.setdefault('verbose', DPEEG_LOGGING_LEVEL)
        loger._update_sh_level(kwargs['verbose'])
        # # for debug
        # print(kwargs, func)
        return func(*args, **kwargs)
    return inner


def set_log_level(
    verbose : Union[int, str] = DPEEG_LOGGING_LEVEL,
    retOldLevel : bool = False
) -> Union[None, str]:
    '''Set the global logging level.
    
    Parameters
    ----------
    verbose : int, str, optional
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for convenience
        and are equivalent to passing in logging.DEBUG, etc. If None, the 
        environment variable `DPEEG_LOGGING_LEVEL` is read, defaults to INFO.
    retOldLevel : bool, optional
        If True, return the old verbosity level. Default is False.

    Returns
    -------
    The old level. Only returned if `retOldLevel` is True.
    '''
    global DPEEG_LOGGING_LEVEL
    oldLevel = loger._get_sh_level()
    DPEEG_LOGGING_LEVEL = verbose
    mne.set_log_level(verbose)
    if retOldLevel:
        return oldLevel


def _set_random_seed(seed : int) -> None:
    '''Set the seed for Python's built-in random module and numpy.

    Parameters
    ----------
    seed: int
        The random seed to use.
    '''
    random.seed(seed)
    np.random.seed(seed)


def _set_torch_seed(seed : int) -> None:
    '''Set the seed for Python's built-in random module and numpy.

    Parameters
    ----------
    seed: int
        The random seed to use.

    '''
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_seed(
    seed : int = DPEEG_SEED, 
    retOldSeed : bool = False
) -> Union[None, int]:
    '''Set the seed for random, numpy and PyTorch.

    Parameters
    ----------
    seed : int, optional
        The random seed to use. If None, the environment variable `DPEEG_SEED` 
        is read, defaults to 42.
    retOldSeed : bool, optional
        If True, return the old seed. Default is False.

    Returns
    -------
    The old seed. Only returned if `retOldSeed` is True.
    '''
    global DPEEG_SEED
    oldSeed = DPEEG_SEED
    DPEEG_SEED = seed
    _set_random_seed(seed)
    _set_torch_seed(seed)
    if retOldSeed:
        return oldSeed
    