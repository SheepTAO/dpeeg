import numpy as np
from mne.io import Raw, read_raw_edf

from .base import RawDataset
from .download import data_dl
from ..utils import get_init_args
from ..tools.docs import fill_doc


@fill_doc
class Mumtaz2016(RawDataset):
    """Heavy development."""

    def __init__(
        self,
        subjects: list[int] | None = None,
        tmin: float = 0,
        tmax: float | None = None,
        picks: list[str] | None = None,
        resample: float | None = None,
        rename: str | None = None,
    ) -> None:
        super().__init__(
            repr=get_init_args(self, locals(), rename=rename, ret_dict=True),
            sign="mumtaz2016",
            subject_list=list(range(1, 65)),
            event_id={"major_depressive_disorder": 1, "healthy_controls": 2},
            subjects=subjects,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            resample=resample,
        )
        self._data_url = "doi:10.6084/m9.figshare.4244171.v2"
        self.bad_file = [
            "H12EC",
            "H18EC",
            "H25EO",
            # "MDD1EO",
            "MDD2EC",
            "MDD4EC",
            "MDD7EC",  # ?
            "MDD7EO",
            "MDD8EC",
            "MDD8EO",
        ]

    def _get_subject_raw(
        self, subject: int, verbose="ERROR"
    ) -> dict[str, dict[str, Raw]]:
        sub = subject - 30 if subject > 30 else subject
        sub_type = "MDD" if subject > 30 else "H"

        session = {}
        for task_type in ["EC", "EO"]:
            if f"{sub_type}{sub}{task_type}" in self.bad_file:
                continue

            filename = data_dl(
                f"{self._data_url}/{sub_type} S{sub} {task_type}.edf",
                path=self.get_dataset_path(),
                force_update=False,
            )
            session[task_type] = filename

        return session


class Liu2024(RawDataset):
    pass
