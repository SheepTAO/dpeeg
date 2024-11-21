import numpy as np
from mne.io import read_raw_gdf
from mne.channels import make_standard_montage

from .base import EpochsDataset
from .download import data_dl
from ..utils import get_init_args
from ..tools.docs import fill_doc


@fill_doc
class Ofner2017(EpochsDataset):
    """Upper limb motor imagery dataset from Ofner et al 2017.

    .. admonition:: Dataset summary

        ====== ====== ======= ===== ======== ======
        Subj   Chan   Time    Cls   Freq     Sess
        ====== ====== ======= ===== ======== ======
        15     61     3 s     7     512 Hz   2
        ====== ====== ======= ===== ======== ======

    Upper limb motor imagery dataset from the paper [1]_. Paper recruited 15
    healthy subjects aged between 22 and 40 years with a mean age of 27 years
    (standard deviation 5 years). Nine subjects were female, and all the
    subjects except s1 were right-handed.

    Paper measured each subject in two sessions on two different days, which
    were not separated by more than one week. In the first session the subjects
    performed ME, and MI in the second session. The subjects performed six
    movement types which were the same in both sessions and comprised of
    elbow flexion/extension, forearm supination/pronation and hand open/close;
    all with the right upper limb. All movements started at a
    neutral position: the hand half open, the lower arm extended to 120
    degree and in a neutral rotation, i.e. thumb on the inner side.
    Additionally to the movement classes, a rest class was recorded in which
    subjects were instructed to avoid any movement and to stay in the starting
    position. In the ME session, the subjects were instructed to execute
    sustained movements. In the MI session, the subjects were asked to perform
    kinesthetic MI of the movements done in the ME session (subjects performed
    one ME run immediately before the MI session to support kinesthetic MI).

    The paradigm was trial-based and cues were displayed on a computer screen
    in front of the subjects. At second 0, a beep sounded and a cross popped up
    on the computer screen (subjects were instructed to fixate their gaze on
    the cross). Afterwards, at second 2, a cue was presented on the computer
    screen, indicating the required task (one out of six movements or rest) to
    the subjects. At the end of the trial, subjects moved back to the starting
    position. In every session, we recorded 10 runs with 42 trials per run. We
    presented 6 movement classes and a rest class and recorded 60 trials per
    class in a session.

    References
    ----------
    .. [1] Ofner, P., Schwarz, A., Pereira, J. and Müller-Putz, G.R., 2017.
           Upper limb movements can be decoded from the time-domain of
           low-frequency EEG. PloS one, 12(8), p.e0182578.
           https://doi.org/10.1371/journal.pone.0182578

    Parameters
    ----------
    %(subjects)s
    %(epochs_tmin_tmax)s
    %(baseline)s
    %(picks)s
    %(resample)s
    %(imagined)s
    %(executed)s
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
        imagined: bool = True,
        executed: bool = False,
        rename: str | None = None,
    ) -> None:
        if (not imagined) and (not executed):
            raise AttributeError("At least one of imagined and executed is True")

        super().__init__(
            repr=get_init_args(self, locals(), rename=rename, ret_dict=True),
            sign="ofner2017",
            subject_list=list(range(1, 16)),
            interval=[0, 3],
            event_id={
                "elbow_flexion": 1536,
                "elbow_extension": 1537,
                "supination": 1538,
                "pronation": 1539,
                "hand_close": 1540,
                "hand_open": 1541,
                "rest": 1542,
            },
            subjects=subjects,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            resample=resample,
            unit_factor=1,
        )
        self._data_url = "doi:10.5281/zenodo.834976"

        self._eog = ["eog-l", "eog-m", "eog-r"]
        self._montage = make_standard_montage("standard_1005")

        self._sessions = []
        if imagined:
            self._sessions.append((1, "imagination"))
        if executed:
            self._sessions.append((0, "execution"))

    def _get_subject_raw(self, subject: int, verbose="ERROR"):
        sessions = {}
        for sess, session in self._sessions:
            data = {}
            for run in range(1, 11):
                filename = data_dl(
                    f"{self._data_url}/motor{session}_subject{subject}_run{run}.gdf",
                    path=self.get_dataset_path(),
                    force_update=False,
                )

                raw = read_raw_gdf(
                    filename,
                    eog=self._eog,
                    misc=range(64, 96),
                    preload=True,
                    verbose=verbose,
                )
                raw.set_montage(self._montage)
                # there is nan in the data
                raw._data[np.isnan(raw._data)] = 0  # type: ignore
                stim = raw.annotations.description.astype(np.dtype("<21U"))
                stim[stim == "1536"] = "elbow_flexion"
                stim[stim == "1537"] = "elbow_extension"
                stim[stim == "1538"] = "supination"
                stim[stim == "1539"] = "pronation"
                stim[stim == "1540"] = "hand_close"
                stim[stim == "1541"] = "hand_open"
                stim[stim == "1542"] = "rest"
                raw.annotations.description = stim

                data[f"run_{run}"] = raw
            sessions[f"session_{sess}"] = data

        return sessions
