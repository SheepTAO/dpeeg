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
import inspect
import functools
import numpy as np
from .tools.logger import Logger
from typing import (
    Any,
    Union,
    Callable,
    TypeVar,
)


DPEEG_SEED : int = 42               # Life, the Universe, and Everything
DPEEG_LOGER_LEVEL = 'INFO'
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
        level = kwargs.get('verbose', DPEEG_LOGER_LEVEL)
        if level:
            loger.update_sh_level(level)
        # for debug
        # print(kwargs['verbose'], func)
        return func(*args, **kwargs)
    return inner


def set_log_level(
    verbose : Union[int, str] = DPEEG_LOGER_LEVEL,
    ret_old_level : bool = False
) -> Union[None, str]:
    '''Set the global logging level.
    
    Parameters
    ----------
    verbose : int, str, optional
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for convenience
        and are equivalent to passing in logging.DEBUG, etc. If None, the 
        environment variable `DPEEG_LOGER_LEVEL` is read, defaults to INFO.
    ret_old_level : bool, optional
        If True, return the old verbosity level. Default is False.

    Returns
    -------
    oldLevel : str, None
        The old level. Only returned if `ret_old_level` is True.
    '''
    global DPEEG_LOGER_LEVEL
    oldLevel = loger.get_sh_level()
    DPEEG_LOGER_LEVEL = verbose
    mne.set_log_level(verbose)
    if ret_old_level:
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
    ret_old_seed : bool = False
) -> Union[None, int]:
    '''Set the seed for random, numpy and PyTorch.

    Parameters
    ----------
    seed : int, optional
        The random seed to use. If None, the environment variable `DPEEG_SEED` 
        is read, defaults to 42.
    ret_old_seed : bool, optional
        If True, return the old seed. Default is False.

    Returns
    -------
    The old seed. Only returned if `ret_old_seed` is True.
    '''
    global DPEEG_SEED
    oldSeed = DPEEG_SEED
    DPEEG_SEED = seed
    _set_random_seed(seed)
    _set_torch_seed(seed)
    if ret_old_seed:
        return oldSeed


def unpacked(*args) -> list:
    '''Positional arguments are unpacked as lists.
    '''
    result = [X for X in args]
    return result


def dict_to_str(kwargs : dict, symbol : str = ', ') -> str:
    '''Convert the dictionary into a string format.

    Parameters
    ----------
    kwargs : dict
        The dictionary to be converted.
    symbol : str
        Join all key-value pairs with the specified separator character.
        Default is ', '.
    '''
    s = [f'{k}={v}' for k, v in kwargs.items()]
    return symbol.join(s)


def align_text(prefix : str, text : str, only_head : bool = True) -> str:
    '''Aligns text, adding a specified prefix to the beginning of line.

    Parameters
    ----------
    prefix : str
        The prefix to be added at the beginning of line.
    text : str
        The text to be aligned, including multiple lines.
    only_head : bool
        If false, prefix will be prepended to each line. Otherwise, spaces with
        a length of prefix will be added to other lines except the first line.

    Returns
    -------
    str: The aligned text.
    '''
    lines = text.splitlines()
    aligned_lines = [f'{prefix}{line}' if i == 0 and only_head else 
                f'{" " * len(prefix)}{line}' for i, line in enumerate(lines)]
    return '\n'.join(aligned_lines)


def get_init_args(
    obj,
    locals : dict[str, Any],
    format : str = 'log',
) -> str:
    '''Get object initialization parameters.

    Assemble the parameters passed in by class initialization or function 
    according to the specified format.

    Parameters
    ----------
    obj : class, function
        The kind of class or function.
    locals : dict
        Local parameters in class __init__.
    format : str
        Parameters splicing format. Default is log.

    Returns
    -------
    str : 
        A string specifying the format.
    '''
    sig = inspect.signature(obj).parameters.keys()

    s = str()
    if format == 'log':
        s += f'[{obj.__name__}:\n'
        for key in sig:
            s += align_text(f'  [{key}]: ', str(locals[key])) + '\n'
        s += f']'
    elif format == 'runtime':
        s += f'{obj.__name__}('
        kwargs = {key: locals[key] for key in sig}
        s = s + dict_to_str(kwargs) + ')'
    else:
        raise ValueError(f'{format} format is not supported.')

    return s
