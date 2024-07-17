import abc
from collections.abc import Iterator
from typing import overload
from pathlib import Path


import mne
from tqdm import tqdm
from numpy import ndarray
from mne.io import Raw
from mne import Epochs


from dpeeg.data.loaddata import load_data
from dpeeg.data.transforms import Transforms, Identity
from dpeeg.data.functions import _get_subject_list, load


__all__ = [
    'EEGData',
    'EEGDataset',
    'LoadDataset',
]


class EEGData(dict):
    @overload
    def __init__(
        self,
        edata : ndarray,
        lable : ndarray,
        **kwargs,
    ) -> None:
        '''test
        '''
        pass

    @overload
    def __init__(
        self,
        tr_edata : ndarray,
        tr_label : ndarray,
        te_edata : ndarray,
        te_label : ndarray,
        **kwargs,
    ) -> None:
        '''test
        '''
        pass

    def __init__(self, *args, **kwargs):
        if len(args) == 4:
            tr_edata, tr_label, te_edata, te_label = args
            super().__init__({
                'train': {'edata': tr_edata, 'label': tr_label},
                'test' : {'edata': te_edata, 'label': te_label},
            })
        elif len(args) == 2:
            edata, label = args
            super().__init__({'edata': edata, 'label': label})
        else:
            pass


class EEGDataset(metaclass=abc.ABCMeta):
    '''Base EEG dataset.
    '''
    def __init__(
        self,
        eegdataset : dict[int, EEGData] | None,
        event_id : dict[str, int] | None = None,
        transforms : Transforms | None = None,
    ) -> None:
        self.eegdataset = {} if eegdataset is None else eegdataset
        self.event_id = event_id
        self.transforms = Identity if transforms is None else transforms

    def __setitem__(self, subject : int, eegdata : EEGData):
        self.eegdataset[subject] = eegdata

    def __getitem__(self, subject : int) -> EEGData:
        return self._get_single_subject_data(subject)

    def _get_single_subject_data(self, subject : int) -> EEGData:
        return self.transforms({subject: self.eegdataset[subject]})

    def get_data(self, subjects : list[int] | None = None):
        subject_items = self.subject_list if subjects is None else subjects
        
        data = {}
        for subject in subject_items:
            data[subject] = self._get_single_subject_data(subject)
        return data

    def keys(self):
        return self.eegdataset.keys()

    def values(self):
        return self.eegdataset.values()

    def items(self) -> Iterator[tuple[int, dict[int, EEGData]]]:
        for subject in self.keys():
            subject_data = self._get_single_subject_data(subject)
            yield subject, subject_data


class LoadDataset(EEGDataset):
    def __init__(
        self,
        folder : str,
        subjects : list[int] | None = None,
        transforms : Transforms | None = None,
        rename : str | None = None,
    ) -> None:
        self.folder = folder
        self.subjects = _get_subject_list(folder, subjects, verbose=False)
        self.transforms = Identity() if transforms is None else transforms

    def _get_single_subject_data(self, subject: int) -> EEGData:
        return self.transforms(
            load(self.folder, self.subjects, False, False)
        )[subject]
    
    def get_data(self, progressbar : bool = True) -> dict[int, EEGData]:
        subjects = tqdm(
            self.subjects, 'Load EEGData', unit='sub', disable=not progressbar
        )

        data = {}
        for subject in subjects:
            data[subject] = self._get_single_subject_data(subject)
        return data
    

class MIDataset(EEGDataset):
    _unit_factor = 1e6

    def __init__(
        self,
        subjects : list[int] | None = None,
        tmin : float = 0.0,
        tmax : float | None = None,
        baseline : tuple[int, int] | None = None,
        picks : list[str] | None = None,
        resample : float | None = None,
        transforms : Transforms | None = None,
        preload : bool = False
    ) -> None:
        if tmax is not None and tmin >= tmax:
            raise ValueError('tmax must be greater than tmin')

        self.subjects = self._subject_list if subjects is None else subjects
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.picks = picks
        self.resample = resample
        self.transforms = Identity if transforms is None else transforms
        self.preload = preload

    def _get_single_subject_raw(self, subject: int)->dict[str, dict[str, Raw]]:
        return load_data(subject, dataset=self._code)

    def get_raw(self, progressbar : bool = True) \
        -> dict[int, dict[str, dict[str, Raw]]]:
        '''Return the raw correspoonding to a list of subjects.

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
        '''
        subjects = tqdm(
            self.subjects, 'Load Raw', unit='sub', disable=not progressbar
        )

        data = {}
        for subject in subjects:
            data[subject] = self._get_single_subject_raw(subject)
        return data

    def _get_single_subject_epochs(self, subject : int) \
        ->dict[int, dict[str, Epochs]]:
        data = {}
        for session, runs in self._get_single_subject_raw(subject).items():
            epochs = []
            for run, raw in runs.items():
                proc = self._epochs_from_raw(raw)
                if proc is None:
                    # if the run did not contain any selected event go to next
                    continue
                
                epochs.append(proc)
            data[session] = mne.concatenate_epochs(epochs, verbose=False)
        return data

    def get_epochs(self, progressbar : bool = True) \
        -> dict[int, dict[str, Epochs]]:
        '''Return the epochs correspoonding to a list of subjects.

        The returned data is a dictionary with the following structure:

            data = {'subject_id' :
                        {'session_id' : Epochs}
                    }

        '''
        subjects = tqdm(
            self.subjects, 'Load Epochs', unit='sub', disable=not progressbar
        )

        data = {}
        for subject in subjects:
            data[subject] = self._get_single_subject_epochs(subject)
        return data

    def _get_single_subject_data(self, subject: int) -> dict[int, EEGData]:
        data = {}
        for session,epochs in self._get_single_subject_epochs(subject).items():
            data[session] = self.transforms({
                subject: self._eegdata_from_epochs(epochs)
            })[subject]
        return data
    
    def get_data(self, progressbar : bool = True) -> dict[int, EEGData]:
        '''Return the data correspoonding to a list of subjects.

        The returned data is a dictionary with the following structure:

            data = {'subject_id' :
                        {'session_id' : EEGData}
                    }
        '''
        subjects = tqdm(
            self.subjects, 'Load EEGData', unit='sub', disable=not progressbar
        )

        data = {}
        for subject in subjects:
            data[subject] = self._get_single_subject_data(subject)
        return data

    def _epochs_from_raw(self, raw : Raw) -> Epochs:
        # picks channels
        if self.picks is None:
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
        else:
            picks = mne.pick_channels(
                raw.info['ch_names'], include=self.picks, ordered=True
            )

        # get interval
        tmin = self.tmin + self._interval[0]
        if self.tmax is None:
            tmax = self._interval[1]
        else:
            tmax = self.tmax + self._interval[0]

        events = self._events_from_raw(raw)

        # epoch data
        baseline = self.baseline
        if baseline is not None:
            baseline = (
                self.baseline[0] + self._interval[0],
                self.baseline[1] + self._interval[1],
            )
            bmin = baseline[0] if baseline[0] < tmin else tmin
            bmax = baseline[1] if baseline[1] > tmax else tmax
        else:
            bmin = tmin
            bmax = tmax
        epochs = mne.Epochs(
            raw,
            events,
            event_id=self._event_id,
            tmin=bmin,
            tmax=bmax,
            proj=False,
            baseline=baseline,
            preload=True,
            picks=picks,
            event_repeated='drop',
            verbose=False,
        )
        if bmin < tmin or bmax > tmax:
            epochs.crop(tmin=tmin, tmax=tmax, include_tmax=False)
        if self.resample is not None:
            epochs = epochs.resample(self.resample, verbose=False)
        return epochs
    
    def _eegdata_from_epochs(self, epochs : Epochs) -> EEGData:
        edata = self._unit_factor * epochs.get_data()
        label = epochs.events[:, -1]
        return EEGData(edata, label)

    def _events_from_raw(self, raw : Raw):
        stim_channels = mne.utils._get_stim_channel(
            None, raw.info, raise_error=False
        )
        if len(stim_channels) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            events, _ = mne.events_from_annotations(
                raw, event_id=self._event_id, verbose=False
            )
            offset = int(self._interval[0] * raw.info["sfreq"])
            events[:, 0] -= offset  # return the original events onset

        return events