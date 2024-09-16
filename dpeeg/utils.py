# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

import os
import random
import inspect
from collections.abc import Iterable, Mapping
from typing import overload, Literal

import mne
import torch
import numpy as np


DPEEG_SEED: int = 42  # Life, the Universe and Everything
DPEEG_DIR = os.path.join(os.path.expanduser("~"), "dpeeg")


def set_log_level(verbose=None, return_old_level=False):
    """Set the logging level.

    Parameters
    ----------
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
        If None, the environment variable MNE_LOGGING_LEVEL is read, and if
        it doesn't exist, defaults to INFO.
    return_old_level : bool
        If True, return the old verbosity level.

    Returns
    -------
    old_level : int
        The old level. Only returned if ``return_old_level`` is True.
    """
    return mne.utils.set_log_level(verbose, return_old_level)


def _set_random_seed(seed: int) -> None:
    """Set the seed for Python's built-in random module and numpy.

    Parameters
    ----------
    seed: int
        The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)


def _set_torch_seed(seed: int) -> None:
    """Set the seed for Python's built-in random module and numpy.

    Parameters
    ----------
    seed: int
        The random seed to use.

    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_seed(seed: int = DPEEG_SEED, ret_old_seed: bool = False) -> None | int:
    """Set global random seed for random, Numpy and PyTorch.

    Parameters
    ----------
    seed : int
        The random seed to use. If None, the environment variable `DPEEG_SEED`
        is read, defaults to 42.
    ret_old_seed : bool, optional
        If True, return the old seed.

    Returns
    -------
    old_seed : int, None
        The old seed. Only returned if `ret_old_seed` is True.
    """
    global DPEEG_SEED
    old_seed = DPEEG_SEED
    DPEEG_SEED = seed
    _set_random_seed(seed)
    _set_torch_seed(seed)
    if ret_old_seed:
        return old_seed


def unpacked(*args) -> list:
    """Positional arguments are unpacked as lists."""
    result = [X for X in args]
    return result


def mapping_to_str(mapping: Mapping, kv_symbol: str = "=", symbol: str = ", ") -> str:
    """Convert a mapping (e.g., dict) into a string format.

    Parameters
    ----------
    mapping : Mapping
        The mapping (e.g., dict) to be converted into a string.
        It should contain keys and values that can be converted to strings.
    kv_symbol : str
        The separator to be used between key and value in the resulting string.
    symbol : str
        The separator to be used between key-value pairs in the resulting
        string. Defaults to ", " (a comma followed by a space).
    """
    return symbol.join(f"{k}{kv_symbol}{v}" for k, v in mapping.items())


def iterable_to_str(iterable: Iterable, symbol: str = ", ") -> str:
    """Convert any iterable (e.g., list, tuple, set) into a string format.

    Parameters
    ----------
    iterable : Iterable
        The iterable to be converted into a string.
        It should contain elements that can be converted to strings.
    symbol : str
        The separator to be used between elements in the resulting string.
        Defaults to ", " (a comma followed by a space).
    """
    return symbol.join(map(str, iterable))


def align_text(prefix: str, text: str, only_head: bool = True) -> str:
    """Aligns text, adding a specified prefix to the beginning of line.

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
    align_text : str
        The aligned text.

    Examples
    --------
    >>> print(align_text("+ ", text, only_head=False))
    + First line
    + Second line
    """
    return "\n".join(
        [
            (
                f"{prefix}{line}"
                if (i == 0) or (not only_head)
                else f'{" " * len(prefix)}{line}'
            )
            for i, line in enumerate(text.splitlines())
        ]
    )


def _format_log_kv(key, value, spaces: int = 2) -> str:
    return f"[{key}:\n{align_text(' ' * spaces, str(value))}\n]"


def _format_log(input: dict, spaces: int = 2) -> str:
    formatted_items = "\n".join(
        align_text(f"{" " * spaces}[{key}]: ", str(value))
        for key, value in input.items()
        if key != "_obj_name"
    )
    return f"[{input['_obj_name']}:\n{formatted_items}\n]"


def _format_rp(input: dict, kv_symbol: str = "=") -> str:
    obj_name = input.pop("_obj_name")
    formatted_args = mapping_to_str(input, kv_symbol=kv_symbol)
    input["_obj_name"] = obj_name
    return f"{obj_name}({formatted_args})"


@overload
def get_init_args(
    obj,
    locals: dict,
    format: str = "log",
    rename: str | None = None,
    ret_dict: Literal[False] = False,
    **kwargs,
) -> str:
    pass


@overload
def get_init_args(
    obj,
    locals: dict,
    format: str = "log",
    rename: str | None = None,
    ret_dict: Literal[True] = True,
    **kwargs,
) -> dict:
    pass


def get_init_args(
    obj,
    locals,
    format="log",
    rename=None,
    ret_dict=False,
    **kwargs,
) -> str | dict:
    """Get object initialization parameters.

    Assemble the parameters passed in by class initialization or function
    according to the specified format. The function can return either the
    assembled string or the unassembled dictionary for updating internal
    parameters.

    Parameters
    ----------
    obj : class, function
        The kind of class or function.
    locals : dict
        Local parameters in class.__init__ or function.
    format : str
        Parameters splicing format.
        - log: [name \\n\\t param1 \\n\\t param2 \\n\\t ...]
        - rp: name(param1, param2, ...)
    rename : str, optional
        User renamed name. If None, the default class or function name is used.
    ret_dict : bool
        If `True`, return the initialization parameters as a dictionary,
        ignoring the `format` parameter. By default, return a string in the
        specified format.
    **kwargs : dict, optional
        Additional parameters for format assembly.

    Returns
    -------
    str
        A string specifying the format.
    dict
        A dictionary containing initialization parameters.

    Examples
    --------
    When you want to get the parameters of a class and its methods. Note: It is
    best to use `self` as the `obj` parameter during initialization and avoid
    using the class to prevent class_name recognition errors of inherited classes.

    >>> class Base:
    ...     def __init__(self, param1, param2=2) -> None:
    ...         print(get_init_args(self, locals(), "rp"))
    ...     def method(self, param3):
    ...         print(get_init_args(self.method, locals(), "rp"))
    >>> class Sub(Base):
    ...     pass
    >>> obj_base = Base(1)
    Base(param1=1, param2=2)
    >>> obj_base.method(3)
    Base.method(param3=3)
    >>> obj_sub = Sub(1)
    Sub(param1=1, param2=2)
    >>> obj_sub.method(3)
    Sub.method(param3=3)

    Likewise, functions are supported.

    >>> def func(param1, param2 = 2):
    ...     print(get_init_args(func, locals(), "rp"))
    >>> func(1)
    func(param1=1, param2=2)

    You can also rename these:

    >>> class Base:
    ...     def method(self, param):
    ...         print(get_init_args(self.method, locals(), "rp", "Sub.func"))
    >>> obj = Base()
    >>> obj.method(4)
    Sub.func(param=4)
    """
    format_list = {
        "log": _format_log,
        "rp": _format_rp,
    }

    if inspect.isfunction(obj):
        obj_name = obj.__name__
    elif inspect.ismethod(obj):
        class_name = obj.__self__.__class__.__name__
        method_name = obj.__name__
        obj_name = f"{class_name}.{method_name}"
    elif inspect.isclass(obj):
        obj_name = obj.__name__
    # for `self`
    elif isinstance(obj, object):
        obj_name = obj.__class__.__name__
        obj = obj.__init__
    else:
        raise TypeError(f"Unable to parse {obj}.")

    sig = inspect.signature(obj).parameters.keys()
    ret = {"_obj_name": obj_name if rename is None else rename}
    for key in sig:
        ret[key] = str(locals[key])

    if ret_dict:
        return ret
    else:
        if format not in format_list.keys():
            raise ValueError(f"{format} format is not supported.")

        return format_list[format](ret, **kwargs)
