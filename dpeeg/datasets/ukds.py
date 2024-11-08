import numpy as np
import pandas as pd
from zipfile import ZipFile
from scipy.io import loadmat
from mne.channels import make_standard_montage
from mne import create_info
from mne.io import RawArray

from .base import RawDataset, DATA_PATH
from .download import data_dl
from ..utils import get_init_args
from ..tools.docs import fill_doc


URL = "https://reshare.ukdataservice.ac.uk/"


@fill_doc
class MODMA_128_Resting(RawDataset):
    """Multi-modal Open Dataset for Mental-disorder Analysis, Experimental Data.

    .. admonition:: Dataset summary

        ====== ====== ======= ===== ======== ======
        Subj   Chan   Time    Cls   Freq     Sess
        ====== ====== ======= ===== ======== ======
        53     128    4.5 s   2     250 Hz   1
        ====== ====== ======= ===== ======== ======

    Multi-model open dataset for mental-disorder analysis [1]_. The dataset
    includes data mainly from clinically depressed patients and matching normal
    controls. 53 participants include a total of 24 outpatients (13 males and
    11 females; 16-56-year-old) diagnosed with depression, as well as 29 healthy
    controls (20 males and 9 females; 18-55-year-old) were recruited. No
    experimental material. The participants should keep quiet and close their
    eyes as much as possible. Continuous EEG signals were recorded using a
    128-channel HydroCel Geodesic Sensor Net (Electrical Geodesics Inc., Oregon
    Eugene, USA) and Net Station acquisition software (version 4.5.4). The
    sampling frequency was 250 Hz. All raw electrode signals were referenced
    to the Cz. 5 minutes of eyes-closed resting-state EEG was recorded.
    Participants were required to keep awake and still without any bodily
    movements, including heads or legs, and any unnecessary eye movements,
    saccades, and blinks.

    References
    ----------
    .. [1] A. Seal, R. Bajpai, J. Agnihotri, A. Yazidi, E. Herrera-Viedma, and
        O. Krejcar, DeprNet: A deep convolution neural network framework for
        detecting depression using EEG, IEEE Trans. Instrum. Meas., vol. 70,
        pp. 1-13, 2021, doi: 10.1109/TIM.2021.3053999.

    Parameters
    ----------
    %(subjects)s
    %(raw_tmin_tmax)s
    %(picks)s
    %(resample)s
    %(rename)s
    """

    def __init__(
        self,
        subjects: list[int] | None = None,
        tmin: float = 0,
        tmax: float | None = None,
        picks: list[str] | None = None,
        resample: float | None = None,
        rename: str | None = None,
    ) -> None:
        if picks is None:
            picks = [f"E{i}" for i in range(1, 129)]

        super().__init__(
            repr=get_init_args(self, locals(), rename=rename, ret_dict=True),
            subject_list=list(range(1, 54)),
            event_id={"major_depressive_disorder": 1, "healthy_controls": 2},
            subjects=subjects,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            resample=resample,
        )
        self._data_url = (
            f"{URL}854301/4/854301_EEG_128Channels_Resting_Lanzhou_2015.zip"
        )
        self._data_path = DATA_PATH / "modma"
        self._montage = make_standard_montage("GSN-HydroCel-129")
        self._info = create_info(
            ch_names=self._montage.ch_names, sfreq=250.0, ch_types="eeg"
        )

    def _parse_zip(self):
        path_zip = data_dl(self._data_url, self._data_path)

        # Extract the zip file if it hasn't been extracted yet
        path_folder = self._data_path / "EEG_128channels_resting_lanzhou_2015"
        if not path_folder.exists():
            print("The first read requires decompressing all data.")
            with ZipFile(path_zip, "r") as zipf:
                zipf.extractall(self._data_path)

        return path_folder

    def encoding(self):
        """Return the correspondence between subjects and source files within
        the datasets.
        """
        path_folder = self._parse_zip()
        df = pd.read_excel(
            path_folder
            / "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx",
            usecols="A:K",
        )
        df.index += 1
        return df

    def _get_subject_raw(self, subject: int, verbose="ERROR"):
        path_folder = self._parse_zip()
        path_list = sorted(path_folder.glob("*.mat"))

        path_subject = path_list[subject - 1]
        data = loadmat(path_subject, struct_as_record=False, squeeze_me=True)
        eeg_data = data[list(data.keys())[3]]
        eeg_data *= 1e-9
        raw = RawArray(data=eeg_data, info=self._info, verbose=verbose)
        raw.set_montage(self._montage)

        return {"session_1": {"run_1": raw}}

    def _set_label(self, subject: int):
        encoding = self.encoding()
        subject_type = encoding.loc[subject, "type"]

        if subject_type == "MDD":
            return np.array([1])
        else:
            return np.array([2])
