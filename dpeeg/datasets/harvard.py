from typing import Literal

import mne
import numpy as np
from scipy.io import loadmat
from mne.io import read_raw_eeglab, read_raw_cnt

from .base import EpochsDataset, MultiSessEEGData, EEGData
from .download import data_dl
from ..utils import get_init_args
from ..tools.docs import fill_doc


@fill_doc
class MI_2(EpochsDataset):
    """Motor imagery of different joints from the same limb.

    .. admonition:: Dataset summary

        ====== ====== ======= ===== ======== ======
        Subj   Chan   Time    Cls   Freq     Sess
        ====== ====== ======= ===== ======== ======
        25     62     4 s     3     1000 Hz  19
        ====== ====== ======= ===== ======== ======

    This data set was collected from 25 right-handed healthy subjects (19
    males, 16 females, aged 19-27) without MI-based BCI experience [1]_.
    The subjects sat in a comfortable chair at 1-meter distance in front of a
    computer screen. Each trial (8s) began with a white circle at the center of
    the monitor for 2s. After that, a red circle appeared as a cue for 1s to
    remind the subjects of the following target. The target indication ("Hand"
    or "Elbow") was presented on the screen for 4s. We adopted the kinesthetic,
    but not visual, motor imagery paradigm in our research. The subjects were
    asked to concentrate on performing the indicated motor imagery task while
    avoiding any motion during imagination. After the imagination, "Break" was
    presented for 1s before next trial. Data for each subject were collected on
    the same day using the same cap. The experiments contained 7 sessions,
    involving 5 sessions consisting of 40 trials (20 trials per movement
    imagination) and 2 sessions consisting of 50 trials each for resting state.
    The sequence of two MI tasks was randomized. There were breaks of 5 to 10
    minutes between sessions. Thus, there are totally 300 trials (100 trials
    for each type of mental state) in the dataset for the following study.

    The public dataset consists of three stages of data, namely the raw data,
    the pre-processed data, and the trial data that can be directly used for
    classification, so that different researchers can reuse the dataset
    according to their needs.

    - Raw Data:
        Raw EEG data were EEG data were acquired using a 64-channel gel
        electrode cap (according to the standard 10/20 System) with a Neuroscan
        SynAmps2 amplifier (Neuroscan, Inc.). The sampling frequency was
        1000 Hz. The left mastoid reference was used for EEG measurement.
        Electrode impedances were kept below 10 kΩ during the experiment. The
        band-pass filtering range of the system was 0.5-100 Hz. Besides, an
        additional 50 Hz notch filter was used for data acquisition.
    - Processed Data:
        The pre-processing of the collected data was done using the EEGLAB
        toolkit (v14.1.1_b) of MATLAB (R2015a) software. We used Common Average
        Reference (CAR) to spatially filter the data, and performed time-domain
        filtering on the data from 0.1 to 100 Hz. A plugin in EEGLAB called
        Automatic Artifact Removal toolbox (AAR) was used to automatically
        remove the ocular and muscular artifacts in EEG. Note that at this
        stage the two EMG channels that were not relevant to the EEG data
        analysis were removed.
    - Trial Data:
        To facilitate the decoding of movement intentions, the trial data of
        each subject were integrated into a single file, which are ready for
        feature extraction and classification. The data were downsampled to
        200 Hz to reduce computational cost, and performed time-domain
        filtering on the data from 0 to 40 Hz.

    References
    ----------
    .. [1] Ma, X., Qiu, S. & He, H. Multi-channel eeg recording during motor
           imagery of different joints from the same limb. Harvard Dataverse
           https://doi.org/10.7910/DVN/RBN3XG (2020).

    Parameters
    ----------
    %(subjects)s
    %(epochs_tmin_tmax)s
    %(baseline)s
    %(picks)s
    %(resample)s
    stage : str
        Use data from different stages. When using the trial stage,
        :meth:`~MI_2.get_data` is used.
    %(rename)s
    """

    def __init__(
        self,
        subjects: list[int] | None = None,
        tmin: float = 0,
        tmax: float | None = None,
        baseline: tuple[int, int] | None = None,
        picks: list[str] | None = None,
        resample: float | None = None,
        stage: Literal["raw", "proc", "trial"] = "trial",
        rename: str | None = None,
    ) -> None:
        super().__init__(
            repr=get_init_args(self, locals(), rename=rename, ret_dict=True),
            sign="mi_2",
            subject_list=list(range(1, 26)),
            interval=[0, 4],
            event_id={"rest": 0, "hand": 1, "elbow": 2},
            subjects=subjects,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            resample=resample,
        )
        self._data_url = "doi:10.7910/DVN/RBN3XG/"
        self.stage = stage

        # Generate rest ``annotations`` according to its `MI2_saveMat.m`
        self._rest_annotations = mne.Annotations(
            np.linspace(0, 4.25 * 74, 75),
            duration=0,
            description="rest",
        )

    def _reset_annotations(self, raw, task_type):
        if task_type == "motorimagery":
            raw.annotations.rename({"1": "hand", "2": "elbow"})
        else:
            raw.set_annotations(self._rest_annotations)
        return raw

    def _get_subject_stage(self, subject: int, verbose="ERROR"):
        if self.stage == "raw":
            sessions = {}
            for session in range(1, 20):
                url = f"{self._data_url}sub-{subject:03d}_ses-{session:02d}_task"
                path = self.get_dataset_path() / "sourcedata" / f"sub{subject:03d}"
                task_type = "motorimagery" if session < 16 else "rest"
                filename = data_dl(f"{url}-{task_type}_eeg.cnt", path, False)

                raw = read_raw_cnt(
                    filename,
                    eog=["HEO", "VEO"],
                    emg=["EMG1", "EMG2"],
                    preload=True,
                    verbose=verbose,
                )
                raw = self._reset_annotations(raw, task_type)
                sessions[f"session_{session}"] = {"run_1": raw}

        elif self.stage == "proc":
            sessions = {}
            for session in range(1, 20):
                url = f"{self._data_url}sub-{subject:03d}_ses-{session:02d}_task"
                path = self.get_dataset_path() / f"sub{subject:03d}"

                task_type = "motorimagery" if session < 16 else "rest"
                file_ext = [
                    "eeg.fdt",
                    "eeg.set",
                    "events.tab" if task_type == "motorimagery" else None,
                ]
                downloads = [f"{url}-{task_type}_{ext}" for ext in file_ext if ext]
                filenames = [data_dl(download, path, False) for download in downloads]

                raw = read_raw_eeglab(
                    filenames[1],
                    preload=True,
                    uint16_codec="latin1",
                    eog=["HEO", "VEO"],
                    verbose=verbose,
                )
                raw = self._reset_annotations(raw, task_type)

                sessions[f"session_{session}"] = {"run_1": raw}

        elif self.stage == "trial":
            filename = data_dl(
                f"{self._data_url}sub-{subject:03d}_task-motorimagery_eeg.mat",
                path=self.get_dataset_path() / "derivatives",
                force_update=False,
            )
            data = loadmat(filename)

            sessions = MultiSessEEGData({})
            for task in range(15):
                sessions[f"task_{task+1}"] = EEGData(
                    edata=data["task_data"][task], label=data["task_label"][task]
                )
            sessions[f"rest"] = EEGData(edata=data["rest_data"], label=np.zeros(300))

        else:
            raise ValueError(
                f"stage should be `raw`, `proc` or `trial`, but got {self.stage}"
            )

        return sessions

    def _get_subject_raw(self, subject: int, verbose="ERROR"):
        if self.stage == "trial":
            return {}
        else:
            return self._get_subject_stage(subject, verbose)

    def _get_subject_data(self, subject: int, verbose="ERROR"):
        if self.stage == "trial":
            return self._get_subject_stage(subject, verbose)
        else:
            return super()._get_subject_data(subject, verbose)
