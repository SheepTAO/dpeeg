# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import sys

from mne.utils._bunch import BunchConst


##############################################################################
# Define our standard documentation entries
#
# To reduce redundancy across functions, please standardize the format to
# ``argument_optional_keywords``. For example ``tmin_raw`` for an entry that
# is specific to ``raw`` and since ``tmin`` is used other places, needs to
# be disambiguated. This way the entries will be easy to find since they
# are alphabetized (you can look up by the name of the argument). This way
# the same ``docdict`` entries are easier to reuse.

docdict = BunchConst()

# fmt: off

# %%
# B

docdict["baseline"] = """
baseline : tuple of int, optional
    The time interval to consider as “baseline” when applying baseline
    correction. If ``None``, do not apply baseline correction.
"""

# %%
# N

docdict["nCh"] = """
nCh : int
    Number of electrode channels.
"""

docdict["nTime"] = """
nTime : int
    Number of data sampling points. For example, a 4-second data input with a
    sampling rate of 250 Hz is 1000.
"""

docdict["nCls"] = """
nCls : int
    Number of classification categories.
"""

# %%
# O

docdict["out_folder"] = """
out_folder : str, optional
    Store all experimental results in a folder named with the model class name 
    in the specified folder ('~/dpeeg/out/model/exp/dataset/timestamp').
"""

# %%
# P

docdict["picks"] = """
picks : list of str, optional
    Channels to include. If ``None``, pick all channels.
"""

# %%
# R

docdict["resample"] = """
resample : float, optional
    Resample data.
"""

docdict["rename"] = """
rename : str, optional
    Rename the dataset.
"""

# %%
# S

docdict["subjects"] = """
subjects : list of int, optional
    List of subject number. If ``None``, all subjects will be loaded.
"""

# %%
# T

docdict["epochs_tmin_tmax"] = """
tmin-tmax : float
    Start and end time of the epochs in seconds, relative to the time locked 
    event. The closest or matching samples corresponding to the start and end 
    time are included. Default is start and end time of epochs, respectively.
"""

docdict["trainer"] = """
trainer : Trainer
    Trainer used for training module on dataset.
"""

docdict["timestamp"] = """
timestamp : bool
    Output folders are timestamped.
"""

# fmt: on

docdict_indented = {}


def fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.
    """
    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = " " * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = "\n".join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError(f"Error documenting {funcname}:\n{exp}")
    return f


def _indentcount_lines(lines):
    """Compute minimum indent for all lines in line list."""
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno
