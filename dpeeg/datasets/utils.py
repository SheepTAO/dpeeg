# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

import sys
import json
from pathlib import Path
from collections.abc import Iterable
from typing import TypeVar

import numpy as np
from numpy.lib import format
from mne.utils import verbose, logger
from tqdm import tqdm

from .. import __version__
from ..utils import iterable_to_str
from .base import BaseData, BaseDataset, EEGData, MultiSessEEGData, SplitEEGData


CURRENT_MODULE = sys.modules[__name__]
T = TypeVar("T")


@verbose
def check_inter_and_compl(
    r_set: Iterable[T], q_set: Iterable[T], strict: bool = True, verbose=None
) -> tuple[set[T], set[T]]:
    """Check the intersection and complement of two sets.

    Check whether the search set and query set meet the requirements and return
    the corresponding intersection and complement.

    Parameters
    ----------
    r_set : Iterable of Any
        Retrieve a collection.
    q_set : Iterable of Any
        Query a collection.
    strict : bool
        By default, an error is raised if any element in the retrieved set does
        not exist in the query set. If `False`, the intersection is allowed to
        be empty.

    Returns
    -------
    inter : set
        Retrieves the intersection of a set and a query set.
    compl : set
        Retrieves the complement of a set with respect to a query set.

    Examples
    --------
    >>> r_set, q_set = [1, 2, 3], [2, 3, 4]
    >>> inter, compl = check_inter_and_compl(r_set, q_set)
    Cannot find 1, only use 2, 3.
    >>> inter, compl
    ({2, 3}, {1})
    """
    r_set, q_set = set(r_set), set(q_set)

    inter = r_set & q_set
    if len(inter) == 0:
        if strict:
            raise KeyError(
                f"Cannot find {iterable_to_str(r_set)}, "
                f"data only contains {iterable_to_str(q_set)}."
            )
        else:
            logger.warning(f"Cannot find {iterable_to_str(r_set)}.")

    compl = r_set - q_set
    if len(compl) != 0:
        logger.warning(
            f"Cannot find {iterable_to_str(compl)}, "
            f"only use {iterable_to_str(inter)}."
        )

    return inter, compl


def save(
    file: str | Path,
    eegdata: BaseData,
    compress: bool = False,
):
    """Archive eegdata into a file in '.egd' format.

    Parameters
    ----------
    file : str, Path
        Filename where the eegdata will be saved, a '.egd' extension will be
        appended to the filename if it does not already have one.
    eegdata : eegdata
        An eegdata to be saved.
    compress : bool
        If ``True``, use standard ZIP compression.
    """
    import zipfile
    import json

    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED

    path = Path(file).resolve().with_suffix(".egd")
    with zipfile.ZipFile(path, mode="w", compression=compression) as zipf:
        eegdata_info = {"version": __version__, "type": type(eegdata).__name__}
        zipf.writestr("eegdata_info.json", json.dumps(eegdata_info, indent=4))

        for val, key in eegdata.datas():
            for k, v in val.items():
                fname = k + ".npy" if key is None else Path(key) / (k + ".npy")
                with zipf.open(str(fname), "w", force_zip64=True) as fid:
                    format.write_array(fid, v, allow_pickle=True)


def save_dataset(
    folder: str | Path,
    dataset: BaseDataset,
    compress: bool = False,
    name_folder: bool = True,
    progressbar: bool = True,
):
    """Save the eegdata dataset to a folder.

    The eegdata in the dataset is stored separately by subject, and a `.json`
    formatted file describing the basic information of the dataset is saved.

    Parameters
    ----------
    folder : str, Path
        The folder location where the dataset is saved.
    dataset : eegdata dataset
        An eegdata dataset to be saved.
    compress : bool
        If `True`, each subject's eegdata is saved using standard ZIP
        compression.
    name_folder : bool
        If `True`, a subfolder with the name of the dataset is created under
        the folder `folder` and saved.
    progressbar : bool
        Whether to show the progress bar.
    """
    path = Path(folder).resolve()
    if name_folder:
        path = path / dataset._repr["_obj_name"]

    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"'{str(path)}' is not a empty folder.")

    dataset_info = {
        "version": __version__,
        "name": dataset._repr["_obj_name"],
        "event_id": dataset._repr["event_id"],
    }
    with open(path / "dataset_info.json", "w") as filer:
        json.dump(dataset_info, filer, indent=4)

    subjects = tqdm(
        dataset.items(),
        "Save dataset",
        total=len(dataset),
        unit="sub",
        dynamic_ncols=True,
        disable=not progressbar,
    )

    for sub, sub_data in subjects:
        save(path / f"sub_{sub}", sub_data, compress=compress)


def _load_EEGData(zipf, filelist) -> EEGData:
    data = {}
    for file in filelist:
        with zipf.open(file, "r") as fid:
            data[Path(file).stem] = format.read_array(fid, allow_pickle=True)
    return EEGData(**data)


def _load_two_wrapping_eegdata(zipf, filelist):
    tmp = {}
    for file in filelist:
        sess, fname = Path(file).parts

        with zipf.open(file, "r") as fid:
            _data = format.read_array(fid, allow_pickle=True)
        tmp.setdefault(sess, {}).update({Path(fname).stem: _data})

    data = {}
    for sess, _data in tmp.items():
        data[sess] = EEGData(**_data)

    return data


def _load_MultiSessEEGData(zipf, filelist) -> MultiSessEEGData:
    return MultiSessEEGData(_load_two_wrapping_eegdata(zipf, filelist))


def _load_SplitEEGData(zipf, filelist) -> SplitEEGData:
    return SplitEEGData(**_load_two_wrapping_eegdata(zipf, filelist))


def load(file: str | Path):
    """Load eegdata from `.egd` file."""
    import zipfile
    import json

    path = Path(file).resolve(strict=True)
    with zipfile.ZipFile(path, mode="r") as zipf:
        filelist = zipf.namelist()
        eegdata_info = "eegdata_info.json"

        if eegdata_info not in filelist:
            raise FileNotFoundError(f"eegdata info not found in {path} file.")
        with zipf.open(eegdata_info, "r") as info:
            data = json.loads(info.read())
            eegdata_type = data["type"]
        filelist.remove(eegdata_info)

        return getattr(CURRENT_MODULE, f"_load_{eegdata_type}")(zipf, filelist)
