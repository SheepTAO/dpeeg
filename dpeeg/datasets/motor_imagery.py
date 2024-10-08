# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

import mne
from mne.io import Raw
from mne import Epochs
from tqdm import tqdm

from .base import BaseDataset, EEGData, MultiSessEEGData
from .loaddata import load_data
from ..utils import get_init_args


class MIDataset(BaseDataset):
    """Motor Imagery Dataset."""

    _unit_factor = 1e6

    def _check_attr(self):
        """Check if the subclass attribute exists."""
        self.subject_list = getattr(self, "_subject_list")
        self.code = getattr(self, "_code")
        self.interval = getattr(self, "_interval")
        self.event_id = getattr(self, "_event_id")

    def __init__(
        self,
        subjects: list[int] | None = None,
        tmin: float = 0.0,
        tmax: float | None = None,
        baseline: tuple[int, int] | None = None,
        picks: list[str] | None = None,
        resample: float | None = None,
        rename: str | None = None,
    ) -> None:
        super().__init__(get_init_args(self, locals(), rename=rename, ret_dict=True))
        self._check_attr()

        if tmax is not None and tmin >= tmax:
            raise ValueError("tmax must be greater than tmin")

        self.subjects = self.subject_list if subjects is None else subjects
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.picks = picks
        self.resample = resample

    def keys(self) -> list[int]:
        return self.subjects

    def _get_single_subject_raw(
        self, subject: int, verbose=False
    ) -> dict[str, dict[str, Raw]]:
        return load_data(subject, dataset=self.code)

    def get_raw(
        self, progressbar: bool = True, verbose=False
    ) -> dict[int, dict[str, dict[str, Raw]]]:
        """Return the raw correspoonding to a list of subjects.

        The returned data is a dictionary with the following structure:

            data = {'subject_id':
                        {'session_id':
                            {'run_id': Raw}
                        }
                    }

        subjects are on top, then we have sessions, then runs.
        A session is a recording done in a single day, without removing the EEG
        cap. A session is constitued of at least one run. A run is a single
        contigous recording. Some dataset break session in multiple runs.
        """
        subjects = tqdm(
            self.subjects,
            "Load Raw",
            unit="sub",
            dynamic_ncols=True,
            disable=not progressbar,
        )

        data = {}
        for subject in subjects:
            data[subject] = self._get_single_subject_raw(subject, verbose)
        return data

    def _get_single_subject_epochs(
        self, subject: int, verbose=False
    ) -> dict[str, Epochs]:
        data = {}
        for session, runs in self._get_single_subject_raw(subject, verbose).items():
            epochs = []
            for run, raw in runs.items():
                proc = self._epochs_from_raw(raw)
                if proc is None:
                    # if the run did not contain any selected event go to next
                    continue

                epochs.append(proc)
            data[session] = mne.concatenate_epochs(epochs, verbose=False)
        return data

    def get_epochs(
        self, progressbar: bool = True, verbose=False
    ) -> dict[int, dict[str, Epochs]]:
        """Return the epochs correspoonding to a list of subjects.

        The returned data is a dictionary with the following structure:

            data = {'subject_id' :
                        {'session_id' : Epochs}
                    }

        """
        subjects = tqdm(
            self.subjects,
            "Load Epochs",
            unit="sub",
            dynamic_ncols=True,
            disable=not progressbar,
        )

        data = {}
        for subject in subjects:
            data[subject] = self._get_single_subject_epochs(subject, verbose)
        return data

    def _get_single_subject_data(self, subject: int, verbose=False) -> MultiSessEEGData:
        sessions = self._get_single_subject_epochs(subject, verbose)

        data = []
        for session, epochs in sessions.items():
            data.append(self._data_from_epochs(epochs, verbose))
        return MultiSessEEGData(data)

    def get_data(
        self, progressbar: bool = True, verbose=False
    ) -> dict[int, MultiSessEEGData]:
        """Return the data correspoonding to a list of subjects.

        The returned data is a dictionary with the following structure:

            data = {'subject_id' :
                        {'session_id' : EEGData}
                    }
        """
        subjects = tqdm(
            self.subjects,
            "Load EEGData",
            unit="sub",
            dynamic_ncols=True,
            disable=not progressbar,
        )

        data = {}
        for subject in subjects:
            data[subject] = self._get_single_subject_data(subject, verbose)
        return data

    def _epochs_from_raw(self, raw: Raw, verbose=False) -> Epochs:
        events = self._events_from_raw(raw, verbose)

        # get interval
        tmin = self.tmin + self.interval[0]
        if self.tmax is None:
            tmax = self.interval[1]
        else:
            tmax = self.tmax + self.interval[0]

        # epoch data
        baseline = self.baseline
        if baseline is not None:
            baseline = (
                baseline[0] + self.interval[0],
                baseline[1] + self.interval[1],
            )
            bmin = baseline[0] if baseline[0] < tmin else tmin
            bmax = baseline[1] if baseline[1] > tmax else tmax
        else:
            bmin = tmin
            bmax = tmax
        epochs = mne.Epochs(
            raw,
            events,
            event_id=self.event_id,
            tmin=bmin,
            tmax=bmax,
            proj=False,
            baseline=baseline,
            preload=True,
            event_repeated="drop",
            verbose=verbose,
        )
        if bmin < tmin or bmax > tmax:
            epochs.crop(tmin=tmin, tmax=tmax)
        return epochs.crop(include_tmax=False)

    def _data_from_epochs(self, epochs: Epochs, verbose=False) -> EEGData:
        if self.picks is None:
            picks = mne.pick_types(epochs.info, eeg=True, stim=False)
        else:
            picks = mne.pick_channels(
                epochs.info["ch_names"], include=self.picks, ordered=True
            )
        epochs.pick(picks, verbose=verbose)

        if self.resample is not None:
            epochs.resample(self.resample, verbose=verbose)

        edata = self._unit_factor * epochs.get_data(copy=False)
        label = epochs.events[:, -1]
        return EEGData(edata, label)

    def _events_from_raw(self, raw: Raw, verbose=False):
        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if len(stim_channels) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=verbose)
        else:
            events, _ = mne.events_from_annotations(
                raw, event_id=self.event_id, verbose=verbose
            )
            offset = int(self.interval[0] * raw.info["sfreq"])
            events[:, 0] -= offset  # return the original events onset

        return events


class BCICIV2A(MIDataset):
    """Dataset IIA from BCI Competition IV.

    .. admonition:: Dataset summary

        ====== ====== ====== ===== ======= ======
        Subj   Chan   Time   Cls   Freq    Sess
        ====== ====== ====== ===== ======= ======
        9      22     4 s    4     250 Hz  2
        ====== ====== ====== ===== ======= ======

    This data set consists of EEG data from 9 subjects of a study published in
    [1]_.  The cue-based BCI paradigm consisted of four different motor imagery
    tasks, namely the imagination of movement of the left hand (class 1), right
    hand (class 2), both feet (class 3), and tongue (class 4). Two sessions on
    different days were recorded for each subject.  Each session is comprised
    of 6 runs separated by short breaks. One run consists of 48 trials (12 for
    each of the four possible classes), yielding a total of 288 trials per
    session.

    The subjects were sitting in a comfortable armchair in front of a computer
    screen.  At the beginning of a trial ( t = 0 s), a fixation cross appeared
    on the black screen.  In addition, a short acoustic warning tone was
    presented.  After two seconds ( t = 2 s), a cue in the form of an arrow
    pointing either to the left, right, down or up (corresponding to one of the
    four classes left hand, right hand, foot or tongue) appeared and stayed on
    the screen for 1.25 s.  This prompted the subjects to perform the desired
    motor imagery task.  No feedback was provided.  The subjects were ask to
    carry out the motor imagery task until the fixation cross disappeared from
    the screen at t = 6 s.

    Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5 cm)
    were used to record the EEG; the montage is shown in Figure 3 left. All
    signals were recorded monopolarly with the left mastoid serving as
    reference and the right mastoid as ground. The signals were sampled with.
    250 Hz and bandpass-filtered between 0.5 Hz and 100 Hz. The sensitivity of
    the amplifier was set to 100 μV . An additional 50 Hz notch filter was
    enabled to suppress line noise

    References
    ----------
    .. [1] Tangermann, M., Müller, K.R., Aertsen, A., Birbaumer, N., Braun, C.,
        Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G. and
        Nolte, G., 2012. Review of the BCI competition IV.
        Frontiers in neuroscience, 6, p.55.
    """

    _code = "bciciv2a"
    _interval = [2.0, 6.0]
    _subject_list = list(range(1, 10))
    _event_id = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}


class BCICIV2B(MIDataset):
    """Dataset IIB from BCI Competition IV.

    .. admonition:: Dataset summary

        ====== ====== ======= ===== ======== ======
        Subj   Chan   Time    Cls   Freq     Sess
        ====== ====== ======= ===== ======== ======
        9      3      4.5 s   2     250 Hz   5
        ====== ====== ======= ===== ======== ======


    This data set consists of EEG data from 9 subjects of a study published in
    [1]_. The subjects were right-handed, had normal or corrected-to-normal
    vision and were paid for participating in the experiments.
    All volunteers were sitting in an armchair, watching a flat screen monitor
    placed approximately 1 m away at eye level. For each subject 5 sessions
    are provided, whereby the first two sessions contain training data without
    feedback (screening), and the last three sessions were recorded with
    feedback.

    Three bipolar recordings (C3, Cz, and C4) were recorded with a sampling
    frequency of 250 Hz.They were bandpass- filtered between 0.5 Hz and 100 Hz,
    and a notch filter at 50 Hz was enabled.  The placement of the three
    bipolar recordings (large or small distances, more anterior or posterior)
    were slightly different for each subject (for more details see [1]).
    The electrode position Fz served as EEG ground. In addition to the EEG
    channels, the electrooculogram (EOG) was recorded with three monopolar
    electrodes.

    The cue-based screening paradigm consisted of two classes,
    namely the motor imagery (MI) of left hand (class 1) and right hand
    (class 2).
    Each subject participated in two screening sessions without feedback
    recorded on two different days within two weeks.
    Each session consisted of six runs with ten trials each and two classes of
    imagery.  This resulted in 20 trials per run and 120 trials per session.
    Data of 120 repetitions of each MI class were available for each person in
    total.  Prior to the first motor im- agery training the subject executed
    and imagined different movements for each body part and selected the one
    which they could imagine best (e. g., squeezing a ball or pulling a brake).

    Each trial started with a fixation cross and an additional short acoustic
    warning tone (1 kHz, 70 ms).  Some seconds later a visual cue was presented
    for 1.25 seconds.  Afterwards the subjects had to imagine the corresponding
    hand movement over a period of 4 seconds.  Each trial was followed by a
    short break of at least 1.5 seconds.  A randomized time of up to 1 second
    was added to the break to avoid adaptation

    For the three online feedback sessions four runs with smiley feedback
    were recorded, whereby each run consisted of twenty trials for each type of
    motor imagery.  At the beginning of each trial (second 0) the feedback (a
    gray smiley) was centered on the screen.  At second 2, a short warning beep
    (1 kHz, 70 ms) was given. The cue was presented from second 3 to 7.5. At
    second 7.5 the screen went blank and a random interval between 1.0 and 2.0
    seconds was added to the trial.

    References
    ----------
    .. [1] R. Leeb, F. Lee, C. Keinrath, R. Scherer, H. Bischof,
        G. Pfurtscheller. Brain-computer communication: motivation, aim, and
        impact of exploring a virtual apartment. IEEE Transactions on Neural
        Systems and Rehabilitation Engineering 15, 473-482, 2007.
    """

    _code = "bciciv2b"
    _interval = [3.0, 7.5]
    _subject_list = list(range(1, 10))
    _event_id = {"left_hand": 1, "right_hand": 2}


class OpenBMI_MI(MIDataset):
    """BMI/OpenBMI dataset for MI.

    .. admonition:: Dataset summary

        ====== ====== ======= ===== ======== ======
        Subj   Chan   Time    Cls   Freq     Sess
        ====== ====== ======= ===== ======== ======
        54     62     4.0 s   2     1000 Hz  2
        ====== ====== ======= ===== ======== ======

    This data set consists of EEG data from 54 subjects of a study published in
    [1]_. EEG signals were recorded with a sampling rate of 1,000 Hz and
    collected with 62 Ag/AgCl electrodes. The EEG amplifier used in the
    experiment was a BrainAmp (Brain Products; Munich, Germany). The channels
    were nasion-referenced and grounded to electrode AFz. Additionally, an EMG
    electrode recorded from each flexor digitorum profundus muscle with the
    olecranon used as reference. The impedances of the EEG electrodes were
    maintained below 10 k during the entire experiment.

    MI paradigm The MI paradigm was designed following a well-established
    system protocol. For all blocks, the first 3 s of each trial began with a
    black fixation cross that appeared at the center of the monitor to prepare
    subjects for the MI task. Afterwards, the subject performed the imagery
    task of grasping with the appropriate hand for 4 s when the right or left
    arrow appeared as a visual cue. After each task, the screen remained blank
    for 6 s (± 1.5 s). The experiment consisted of training and test phases;
    each phase had 100 trials with balanced right and left hand imagery tasks.
    During the online test phase, the fixation cross appeared at the center of
    the monitor and moved right or left, according to the real-time classifier
    output of the EEG signal.

    References
    ----------
    .. [1] Lee, M. H., Kwon, O. Y., Kim, Y. J., Kim, H. K., Lee, Y. E.,
        Williamson, J., … Lee, S. W. (2019). EEG dataset and OpenBMI toolbox
        for three BCI paradigms: An investigation into BCI illiteracy.
        GigaScience, 8(5), 1-16. https://doi.org/10.1093/gigascience/giz002
    """

    _code = "openbmi_mi"
    _interval = [0.0, 4.0]
    _subject_list = list(range(1, 55))
    _event_id = {"right_hand": 1, "left_hand": 2}


class HighGamma(MIDataset):
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
    """

    _code = "high_gamma"
    _interval = [0.0, 4.0]
    _subject_list = list(range(1, 15))
    _event_id = {"right_hand": 1, "left_hand": 2, "rest": 3, "feet": 4}
