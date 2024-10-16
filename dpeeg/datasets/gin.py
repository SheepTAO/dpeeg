from mne.channels import make_standard_montage
from mne.io import read_raw_edf

from .base import DATA_PATH, _EEGDataset
from .download import data_dl
from ..utils import get_init_args
from ..tools.docs import fill_doc


URL = "https://web.gin.g-node.org/"


@fill_doc
class HighGamma(_EEGDataset):
    """High-gamma dataset described in Schirrmeister et al. 2017.

    .. admonition:: Dataset summary

        ====== ====== ======= ===== ======== ======
        Subj   Chan   Time    Cls   Freq     Sess
        ====== ====== ======= ===== ======== ======
        14     128    4.0 s   4     500 Hz   2
        ====== ====== ======= ===== ======== ======

    This data set consists of EEG data from 14 subjects of a study published in
    [1]_. High-Gamma Dataset is a 128-electrode dataset (of which we later only
    use 44 sensors covering the motor cortex, obtained from 14 healthy subjects
    (6 female, 2 left-handed, age 27.2 ± 3.6 (mean ± std)) with roughly 1000
    (963.1 ± 150.9, mean ± std) four-second trials of executed movements
    divided into 13 runs per subject. The four classes of movements were
    movements of either the left hand, the right hand, both feet, and rest (no
    movement, but same type of visual cue as for the other classes). The
    training set consists of the approx. 880 trials of all runs except the last
    two runs (as session 1), the test set of the approx. 160 trials of the last
    2 runs (as session 2). This dataset was acquired in an EEG lab optimized
    for non-invasive detection of high- frequency movement-related EEG
    components.

    Depending on the direction of a gray arrow that was shown on black back-
    ground, the subjects had to repetitively clench their toes (downward arrow),
    perform sequential finger-tapping of their left (leftward arrow) or right
    (rightward arrow) hand, or relax (upward arrow). The movements were selected
    to require little proximal muscular activity while still being complex
    enough to keep subjects in- volved. Within the 4-s trials, the subjects
    performed the repetitive movements at their own pace, which had to be
    maintained as long as the arrow was showing. Per run, 80 arrows were
    displayed for 4 s each, with 3 to 4 s of continuous random inter-trial
    interval. The order of presentation was pseudo-randomized, with all four
    arrows being shown every four trials. Ideally 13 runs were performed to
    collect 260 trials of each movement and rest. The stimuli were presented
    and the data recorded with BCI2000. The experiment was approved by the
    ethical committee of the University of Freiburg.

    References
    ----------
    .. [1] Schirrmeister, Robin Tibor, et al. "Deep learning with convolutional
        neural networks for EEG decoding and visualization." Human brain
        mapping 38.11 (2017): 5391-5420.

    Parameters
    ----------
    %(subjects)s
    %(epochs_tmin_tmax)s
    %(baseline)s
    %(picks)s
    %(resample)s
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
        rename: str | None = None,
    ) -> None:
        super().__init__(
            repr=get_init_args(self, locals(), rename=rename, ret_dict=True),
            subject_list=list(range(1, 15)),
            interval=[0.0, 4.0],
            event_id={"right_hand": 1, "left_hand": 2, "rest": 3, "feet": 4},
            subjects=subjects,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            resample=resample,
        )
        self._data_url = f"{URL}robintibor/high-gamma-dataset/raw/master/data/"
        self._data_path = DATA_PATH / "high_gamma"

    def _get_single_subject_raw(self, subject: int, verbose="ERROR"):
        sessions = {}
        montage = make_standard_montage("standard_1005")
        for i, r in enumerate(["train", "test"], start=1):
            filename = data_dl(
                f"{self._data_url}{r}/{subject:d}.edf",
                self._data_path / r,
                force_update=False,
            )
            raw = read_raw_edf(
                filename, infer_types=True, preload=True, verbose=verbose
            )
            raw.set_montage(montage)
            sessions.update({f"session_{i}": {"run_1": raw}})

        return sessions
