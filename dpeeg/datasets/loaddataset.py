# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

import json
from pathlib import Path

from tqdm import tqdm
from mne.utils import verbose, logger

from .base import BaseDataset
from .utils import check_inter_and_compl, load
from ..utils import get_init_args
from ..tools.docs import fill_doc


@fill_doc
class LoadDataset(BaseDataset):
    """Read a dataset from a local folder.

    This will read the dataset that has already been saved from the local
    folder. Use with :class:`dpeeg.datasets.save_dataset`.

    Parameters
    ----------
    folder : str, Path
        The path to the saved dataset.
    %(subjects)s
    %(rename)s
    """

    @verbose
    def __init__(
        self,
        folder: str | Path,
        subjects: list[int] | None = None,
        rename: str | None = None,
        verbose=None,
    ) -> None:
        super().__init__(get_init_args(self, locals(), rename=rename, ret_dict=True))
        self.folder = Path(folder).resolve(strict=True)

        dataset_info_path = self.folder / "dataset_info.json"
        if dataset_info_path.exists():
            with open(dataset_info_path, "r") as filer:
                dataset_info = json.load(filer)
            if rename is None:
                self._repr["_obj_name"] = dataset_info["name"]
            self.event_id = dataset_info["event_id"]

        else:
            logger.warning("'dataset_info' file not found, using default parameters.")
            self.event_id = None

        path_list = [p for p in self.folder.iterdir() if p.name != "dataset_info.json"]
        subject_list = list([int(p.stem.split("_")[1]) for p in path_list])
        subject_list.sort()

        if subjects is None:
            self.subjects = subject_list
        else:
            self.subjects, _ = check_inter_and_compl(
                subjects, subject_list, verbose=verbose
            )

        self._repr["subjects"] = len(self.subjects)

    def _get_single_subject_data(self, subject: int):
        subject_path = self.folder / f"sub_{subject}.egd"
        return load(subject_path)

    def get_data(self, progressbar: bool = True):
        """Returns the eegdata of all subjects."""
        subjects = tqdm(
            self.subjects, "Load EEGData", unit="sub", disable=not progressbar
        )

        data = {}
        for subject in subjects:
            data[subject] = self._get_single_subject_data(subject)
        return data

    def keys(self):
        return self.subjects
