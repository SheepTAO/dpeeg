#!/usr/bin/env python
# coding: utf-8

"""
    This module is used to load commonly used MI-EEG data.
    
    NOTE: All data is read during the initialization phase.
        - Defect: At the beginning, all the data needs to be read once, and the
        initialization time is longer. If the amount of data is huge, the 
        memory consumption is huge.
        - Advantage: In the process of multiple iterations, there is no need to
        read data repeatedly, and only one round of IO operation is required, 
        which saves time for multiple rounds of training.

    NOTE: If you want to build your own data set, you can refer to the sample 
        dataset to build it. Please reassign the `_event_id` and `_epochs` 
        attributes. In the final stage of initialization, please call the 
        `self.load_data()` to completely initialize your dataset.


    @Author  : SheepTAO
    @Time    : 2023-07-26
"""


import os
import mne
import numpy as np
from typing import Literal
from scipy.io import loadmat

from .preprocessing import Preprocess, ComposePreprocess
from .transforms import Transforms, ComposeTransforms, SplitTrainTest
from ..tools.logger import _Level
from ..utils import loger, verbose, DPEEG_SEED, get_init_args
from .functions import load


class EEGDataset:
    '''Base EEG dataset'''
    def _check_attr(self, attr : str) -> dict:
        '''Check if the attribute is None.'''
        if getattr(self, attr) is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} model should implement '{attr}'.")
        return getattr(self, attr)

    @property
    def size(self) -> tuple:
        '''Return the current dimension of the dataset.'''
        sub = next(iter(self.dataset.values()))
        return (1, *sub['train'][0].shape[1:])

    @property
    def event_id(self) -> dict:
        '''Return the event label of the dataset and its corresponding id.'''
        return self._check_attr('_event_id')

    @property
    def cls_name(self) -> tuple:
        '''Return the event label of the dataset.'''
        return tuple(self.event_id.keys())

    @property
    def epochs(self) -> dict:
        '''Return the original Epochs data.'''
        return self._check_attr('_epochs')

    @property
    def dataset(self) -> dict:
        '''Return the loaded data.'''
        return self._check_attr('_dataset')

    def __init__(
        self,
        preprocess : Preprocess | None = None,
        transforms : Transforms | None = None,
        test_size : float = .25, 
        seed : int = DPEEG_SEED,
        verbose : _Level = None, 
    ) -> None:
        '''EEG Dataset abstract base class.

        Parameters
        ----------
        preprocess : Preprocess, optional
            Apply preprocessing on epochs.
        transforms : Transforms, optional
            Apply pre-transforms on dataset.
        test_size : float
            Split the training set and test set proportions. If the dataset is
            already split, it will be ignored.
        seed : int
            Random seed when splitting.
        verbose : int, str, optional
            Log level of mne.
        '''
        mne.set_log_level(verbose)

        self._repr = None
        self._preprocess = preprocess
        self._transforms = transforms
        self._test_size = test_size
        self._seed = seed
        self._dataset = None
        self._verbose = verbose

        # NOTE
        # Please make sure the following attributes are correctly overridden
        self._event_id = None        # task name and its corresponding label
        self._epochs = None         # each subject and its corresponding Epochs
    
    def load_data(
        self, 
        split : bool = False,
        unit_factor : float = 1e6,
        include_tmax : bool = False,
    ) -> None:
        '''Extract data from Epochs and split.

        Parameters
        ----------
        split : bool
            Whether `self.raw` has been splited.
        unit_factor : float
            Unit factor to convert the units of uv to v.
        include_tmax : bool
            Whether to include the last timestamp is used to solve the data 
            timestamp problem after resampling and will be removed in future
            versions.

        Notes
        -----
        Avoid data leakage when you split data.
        '''
        if self._preprocess:
            pres = ComposePreprocess(self._preprocess)
            self._epochs = pres(self.epochs)

        dataset = {}
        for sub, sEpochs in self.epochs.items():
            if not split:
                data = sEpochs.crop(include_tmax=include_tmax).get_data()
                label = sEpochs.events[:, -1]
                dataset[sub] = [data * unit_factor, label]
            else:
                dataset[sub] = {}
                for mode, mEpochs in sEpochs.items():
                    data = mEpochs.crop(include_tmax=include_tmax).get_data()
                    label = mEpochs.events[:, -1]
                    dataset[sub][mode] = [data * unit_factor, label]

        # split the dataset before transforms
        if self._transforms:
            trans = ComposeTransforms(self._transforms)
            if not split:
                trans.insert(0, SplitTrainTest(self._test_size, self._seed))
            self._dataset = trans(dataset, verbose=self._verbose)
        else:
            self._dataset = dataset
        loger.info('[Loading dataset done]')

    def process_interval(
        self,
        tmin : float,
        tmax : float,
        interval : tuple | None = None,
        baseline : tuple | None = None,
    ) -> tuple:
        '''Preprocessing of data extraction time.

        This function is used to correct the extraction timestamps and baseline
        adjustments of different data sets according to the interval. Can be 
        called as needed.

        Parameters
        ----------
        tmin : float
             Start time (in second) of the epoch, relative to the dataset spec-
             ific task interval e.g. tmin = 1 would mean the epoch will start 1
             second after the beginning of the task as defined by the dataset.
        tmax : float, optional
            End time (in second) of the epoch, relative to the beginning of the
            dataset specific task interval. tmax = 5 would mean the epoch will 
            end 5 second after the beginning of the task as defined in the data
            set.
        interval : tuple of length 2, optional
            Imagery interval as defined in the dataset description.
        baseline : tuple of length 2, optional
            The time interval to consider as “baseline” when applying baseline 
            correction. If None, do not apply baseline correction. If a tuple 
            (a, b), the interval is between a and b (in seconds), including the
            endpoints.

        Returns
        -------
        If baseline is not None, will return the corrected `tmin`, `tmax` and 
        `baseline`, or only return the corrected `tmin`, `tmax` and `None`.
        '''
        if tmin >= tmax:
            raise ValueError('tmax must be greater than tmin')

        if interval:
            if len(interval) == 2:
                if interval[0] >= interval[1]:
                    raise ValueError('End time of the interval must be greater'
                                     ' than the start time.')
                tmin, tmax = interval[0] + tmin, interval[0] + tmax

                if baseline:
                    baseline = (
                        baseline[0] + interval[0], baseline[1] + interval[0]
                    )
                    bmin = baseline[0] if baseline[0] < tmin else tmin
                    bmax = baseline[1] if baseline[1] > tmax else tmax
                    return tmin, tmax, (bmin, bmax)
                else:
                    return tmin, tmax, baseline

            else:
                raise ValueError('Interval should be the start and end time '
                                 f'points, but got {len(interval)} numbers.')
        else:
            return tmin, tmax, baseline

    def __getitem__(self, args) -> dict:
        '''Return the index position data of the corresponding subject
        and the corresponding set.

        Parameters
        ----------
        args : sub, index (optional)
            sub : int
                Number of subject.
            index : int, slice, optional
                The data index corresponding to the training set. If index is 
                None, all the data (train and test) of the subject will be re-
                turned as a dictionary.

        Returns
        -------
        dict - A dict contain train set (and test set).

        Example
        -------
            >>> dataset[1]
            {'train': (sub1[data], sub1[label]), 
             'test': (sub1[data], sub1[label])}
            >>> dataset[2, :2]
            (sub2['train'][data[:2]], sub2['train'][label[:2]])
        '''
        if isinstance(args, int):
            sub = args
            index = None
        else:
            num = len(args)
            if num != 2:
                raise ValueError(f'Index must be 1 to 2 numbers, but got {num}.')
            else:
                sub, index = args[0], args[1]

        if index is not None:
            return {
                'train' : (self.dataset[sub]['train'][0][index],
                           self.dataset[sub]['train'][1][index])
            }
        else:
            return {
                'train': tuple(self.dataset[sub]['train']),
                'test': tuple(self.dataset[sub]['test'])
            }

    def __repr__(self) -> str:
        if self._repr:
            return self._repr
        else:
            class_name = self.__class__.__name__
            return f'{class_name} not implement attribute `self._repr`.'

    def items(self):
        return self.dataset.items()

    def keys(self):
        return self.dataset.keys()

    def values(self):
        return self.dataset.values()


class PhysioNet(EEGDataset):
    '''Physionet MI Dataset.
    channels=64; subjects=1-109[88, 92, 100, 104], tasks={0:left hand, 1:right 
    hand, 2:hands, 3:feet}; duration=3s; freq=160Hz, sessions=1.
    '''
    @verbose
    def __init__(
        self,
        subjects : list[int] | None = None,
        tmin : float = 0,
        tmax : float = 1,
        preprocess : Preprocess | None = None,
        transforms : Transforms | None = None,
        test_size : float = .25,
        picks : list[str] | None = None,
        baseline = None,
        seed : int = DPEEG_SEED,
        verbose : _Level = None
    ) -> None:
        '''Physionet MI Dataset.

        Parameters
        ----------
        subjects : list of int, optional
            List of subject number. If None, all subjects will be loaded. 
        tmin, tmax : float
            Start and end time of the epochs in seconds, relative to the time-
            locked event. The closest or matching samples corresponding to the
            start and end time are included. Default is start and end time of 
            epochs, respectively.
        picks : list of str, optional
            Channels to include. If None, pick all channels.
        preprocess : Preprocess, optional
            Apply preprocessing on epochs.
        transforms : Transforms, optional
            Apply pre-transforms on dataset.
        test_size : float
            Split the training set and test set proportions. If the dataset is
            already split, it will be ignored.
        seed : int
            Random seed when splitting.
        '''
        super().__init__(preprocess, transforms, test_size, seed, verbose)
        self._repr = get_init_args(self, locals())
        loger.info('Reading PhysionetMI Dataset ...')

        from moabb.datasets import PhysionetMI
        dataset = PhysionetMI()
        tmin, tmax, baseline = \
            self.process_interval(tmin, tmax, dataset.interval, baseline)
        self._base_raw = dataset.get_data(subjects)
        self.bad_sub = [88, 92, 100, 104]
        self._event_id = {
            'left_hand': 0,
            'right_hand': 1,
            'hands': 2,
            'feet': 3
        }

        self._epochs = {}
        for sub, sess in self._base_raw.copy().items():
            loger.info(f'\nCreate sub{sub} Epochs ...')
            if sub in self.bad_sub:
                continue
            epochs_ses = []
            for raw in sess['session_0'].values():

                ann = mne.events_from_annotations(raw, regexp='^(?!rest$).*$')
                # update events (labels)
                ann[0][:, -1] = ann[0][:, -1]-1 if 'left_hand' in ann[1].keys() \
                    else ann[0][:, -1]+1
                # update event_id
                for key in ann[1].keys():
                    ann[1][key] = self._event_id[key]

                events = ann[0][:-1]
                epochs_ses.append(
                    mne.Epochs(raw, events, ann[1], tmin, tmax, baseline, picks, 
                               preload=True)
                )

            self._epochs[sub] = mne.concatenate_epochs(epochs_ses, verbose='ERROR')

        self.load_data()


class BCICIV2A(EEGDataset):
    '''BCI Competition IV Data sets 2a.
    channels=22; subjects=1-9; tasks={0:left hand, 1:right hand, 2:feet, 3:tongue};
    duration=4s; freq=250Hz; sessions=2.
    '''
    @verbose
    def __init__(
        self,
        subjects : list[int] | None = None,
        tmin : float = 0,
        tmax : float = 4,
        preprocess : Preprocess | None = None,
        transforms : Transforms | None = None,
        test_size : float = .25,
        mode : Literal['single_ses', 'cross_ses', 'mixed_ses'] = 'cross_ses',
        picks : list[str] | None = None,
        baseline = None,
        seed : int = DPEEG_SEED,
        verbose : _Level = None
    ) -> None:
        '''BCI Competition IV Data sets 2a.

        Parameters
        ----------
        subjects : list of int, optional
            List of subject number. If None, all subjects will be loaded. 
        tmin, tmax : float
            Start and end time of the epochs in seconds, relative to the time-
            locked event. The closest or matching samples corresponding to the
            start and end time are included. Default is start and end time of 
            epochs, respectively.
        picks : list of str, optional
            Channels to include. If None, pick all channels.
        preprocess : Preprocess, optional
            Apply preprocessing on epochs.
        transforms : Transforms, optional
            Apply pre-transforms on dataset.
        mode : str, optional
            If mode = 'single_ses', training data and test data will only use
            session 1. If mode = 'cross_ses', training data and test data will
            use session 1 and 2, respectively. If mode = 'mixed_ses', training
            data and test data will use both session 1 and 2.
        test_size : float
            Split the training set and test set proportions. If the dataset is
            already split, it will be ignored.
        seed : int
            Random seed when splitting.
        '''
        super().__init__(preprocess, transforms, test_size, seed, verbose)
        self._repr = get_init_args(self, locals())
        loger.info('Reading BCICIV 2A Dataset ...')

        from moabb.datasets import BNCI2014001
        dataset = BNCI2014001()
        tmin, tmax, baseline = \
            self.process_interval(tmin, tmax, dataset.interval, baseline)
        self._base_raw = dataset.get_data(subjects)
        self._event_id = {
            'left_hand': 0,
            'right_hand': 1,
            'feet': 2,
            'tongue': 3
        }

        self._epochs = {}
        for sub, sessions in self._base_raw.copy().items():
            loger.info(f'Create sub{sub} Epochs ...')
            epochs_ses_one, epochs_ses_two = [], []
            for session, runs in sessions.items():
                for raw in runs.values():

                    events = mne.find_events(raw, 'stim')
                    events[:, -1] -= 1
                    epochs = mne.Epochs(raw, events, self._event_id, tmin, 
                                        tmax, baseline, picks, preload=True)
                    epochs.drop_channels(['stim', 'EOG1', 'EOG2', 'EOG3'])

                    if session == 'session_T':
                        epochs_ses_one.append(epochs)
                    else:
                        epochs_ses_two.append(epochs)

            if mode == 'single_ses':
                self._epochs[sub] = mne.concatenate_epochs(epochs_ses_one)
            elif mode == 'cross_ses':
                self._epochs[sub] = {}
                self._epochs[sub]['train'] = mne.concatenate_epochs(epochs_ses_one)
                self._epochs[sub]['test'] = mne.concatenate_epochs(epochs_ses_two)
            elif mode == 'mixed_ses':
                self._epochs[sub] = mne.concatenate_epochs(
                    epochs_ses_one + epochs_ses_two
                )
            else:
                raise ValueError(f'Mode does not support {mode}.')

        split = True if mode == 'cross_ses' else False
        self.load_data(split)


class BCICIV2B(EEGDataset):
    '''BCI Competition IV Data sets 2b.
    channels=3; subjects=1-9; tasks={0:left hand, 1:right hand}; duration=4s; 
    freq=250Hz; sessions=5.
    '''
    @verbose
    def __init__(
        self,
        subjects : list[int] | None = None,
        tmin : float = 0,
        tmax : float = 4,
        preprocess : Preprocess | None = None,
        transforms : Transforms | None = None,
        test_size : float = .25,
        mode : Literal['single_ses', 'cross_ses', 'mixed_ses'] = 'cross_ses',
        test_sessions : list[int] | None = None,
        picks : list[str] | None = None,
        baseline = None,
        seed : int = DPEEG_SEED,
        verbose : _Level = None,
    ) -> None:
        '''BCI Competition IV Data sets 2b.

        Parameters
        ----------
        subjects : list of int, optional
            List of subject number. If None, all subjects will be loaded. 
        tmin, tmax : float
            Start and end time of the epochs in seconds, relative to the time-
            locked event. The closest or matching samples corresponding to the
            start and end time are included. Default is start and end time of 
            epochs, respectively.
        picks : list of str, optional
            Channels to include. If None, pick all channels. Default is None.
        preprocess : Preprocess, optional
            Apply preprocessing on epochs.
        transforms : Transforms, optional
            Apply pre-transforms on dataset.
        mode : str, optional
            If mode = 'cross_ses', training data and test data will use differ-
            ent sessions respectively. At this time, you need to specify `test_
            ses` to select which sessions are used as the test set. If mode = 
            'mixed_ses', training data and test data will use all sessions. If 
            mode='single_ses', the training and test sets will come from the 
            same session, and `test_ses` specifies the session used.
        test_sessions : list, optional
            Specify the selected session as the test set.
        test_size : float
            Split the training set and test set proportions. If the dataset is
            already split, it will be ignored.
        seed : int
            Random seed when splitting.
        '''
        super().__init__(preprocess, transforms, test_size, seed, verbose)
        self._repr = get_init_args(self, locals())
        loger.info('Reading BCICIV 2B Dataset ...')

        from moabb.datasets import BNCI2014004
        dataset = BNCI2014004()
        tmin, tmax, baseline = \
            self.process_interval(tmin, tmax, dataset.interval, baseline)
        self._base_raw = dataset.get_data(subjects)
        self._event_id ={'left_hand': 0, 'right_hand': 1}

        test_ses = None
        if test_sessions:
            test_ses = [f'session_{ses}' for ses in test_sessions]

        self._epochs = {}
        for sub, sessions in self._base_raw.copy().items():
            loger.info(f'Create sub{sub} Epochs ...')
            epochs_ses_train, epochs_ses_test = [], []
            for session, runs in sessions.items():
                for run, raw in runs.items():

                    events = mne.find_events(raw)
                    events[:, -1] -= 1

                    epochs = mne.Epochs(raw, events, self._event_id, tmin, 
                                        tmax, baseline, picks, preload=True)
                    epochs.drop_channels(['stim', 'EOG1', 'EOG2', 'EOG3'])
                    print(f'sub{sub}-{session}-{run}: {len(events)}')

                    if mode in ['single_ses', 'cross_ses']:
                        assert test_ses, f'In {mode} mode, test_sessions cannot be None.'
                        if session in test_ses:
                            epochs_ses_test.append(epochs)
                        else:
                            epochs_ses_train.append(epochs)

            if mode == 'single_ses':
                self._epochs[sub] = epochs_ses_test[0]
            elif mode == 'cross_ses':
                self._epochs[sub] = {
                    'train': mne.concatenate_epochs(epochs_ses_train),
                    'test': mne.concatenate_epochs(epochs_ses_test)
                }
            elif mode == 'mixed_ses':
                self._epochs[sub] = mne.concatenate_epochs(
                    epochs_ses_test + epochs_ses_train
                )
            else:
                raise ValueError(f'Mode does not support {mode}.')

        split = True if mode == 'cross_ses' else False
        self.load_data(split=split)


class HGD(EEGDataset):
    '''High Gamma Dataset.
    channels=128; subjects=1-14; tasks={0:feet, 1:left hand, 2:rest, 3:right hand};
    duration=4s; freq=500Hz; sessions=1.

    The dataset per subject included approximately 1040 trials over 13 runs. 
    The first 11 runs (approximately 880 trials) were used for the training 
    and the last 2 runs (approximately 160 trials) were used for evaluation.
    '''
    @verbose
    def __init__(
      self,
      subjects : list[int] | None = None,
      tmin : float = 0,
      tmax : float = 4,
      preprocess : Preprocess | None = None,
      transforms : Transforms | None = None,
      test_size : float = .25,
      picks : list[str] | None = None,
      baseline = None,
      seed : int = DPEEG_SEED,
      verbose : _Level = None,
    ) -> None:
        '''High Gamma Dataset.

        Parameters
        ----------
        subjects : list of int, optional
            List of subject number. If None, all subjects will be loaded. 
        tmin, tmax : float
            Start and end time of the epochs in seconds, relative to the time-
            locked event. The closest or matching samples corresponding to the
            start and end time are included. Default is start and end time of 
            epochs, respectively.
        picks : list of str, optional
            Channels to include. If None, pick all channels.
        preprocess : Preprocess, optional
            Apply preprocessing on epochs.
        transforms : Transforms, optional
            Apply pre-transforms on dataset.
        test_size : float
            Split the training set and test set proportions. If the dataset is
            already split, it will be ignored.
        seed : int
            Random seed when splitting.
        '''
        super().__init__(preprocess, transforms, test_size, seed, verbose)
        self._repr = get_init_args(self, locals())
        loger.info('Reading High Gamma Dataset ...')

        from moabb.datasets import Schirrmeister2017
        dataset = Schirrmeister2017()
        tmin, tmax, baseline = \
            self.process_interval(tmin, tmax, dataset.interval, baseline)
        self._base_raw = dataset.get_data(subjects)
        self._event_id = {
            'feet': 0, 
            'left_hand': 1, 
            'rest': 2, 
            'right_hand': 3
        }

        self._epochs = {}
        for sub, sess in self._base_raw.items():
            loger.info(f'Create sub{sub} Epochs ...')
            for mode, run in sess['session_0'].items():
                events, _ = mne.events_from_annotations(run)
                # update events
                events[:, -1] -= 1
                epochs = mne.Epochs(run, events, self._event_id, tmin, tmax,
                                    baseline, picks, preload=True)
                self._epochs.setdefault(sub, {})[mode] = epochs

        self.load_data(split=True, include_tmax=True)
        

class SEED(EEGDataset):
    @verbose
    def __init__(
      self,
      path : str,
      session : int = 0,
      transforms : Transforms | None = None,
      test_size : float = .25,
      seed : int = DPEEG_SEED,
      verbose : _Level = None,
    ) -> None:
        super().__init__(None, transforms, test_size, seed, verbose)
        self._repr = get_init_args(self, locals())
        loger.info('Reading SEED Dataset ...')

        self._event_id = {
            'negative': 0,
            'netural': 1,
            'positive': 2
        }

        subjects_list = os.listdir(os.path.abspath(path))
        subjects_list.remove('label.mat')
        subjects_list.remove('readme.txt')
        subjects_list.sort(key=lambda x : int(x.split('_')[0]))

        sub_session = [subjects_list[s] for s in range(session, 45, 3)]
        sub_session_path = [os.path.join(path, p) for p in sub_session]

        label_path = os.path.join(path, 'label')
        label = loadmat(label_path)
        label = label['label'].squeeze(0) + 1

        dataset = {}
        for sub, data_path in enumerate(sub_session_path):
            all_trial = loadmat(data_path)
            all_trial.pop('__header__')
            all_trial.pop('__version__')
            all_trial.pop('__globals__')

            trial_list = []
            for trial in all_trial.values():
                trial_list.append(np.expand_dims(trial[:, :36000], 0))
            data = np.concatenate(trial_list)
            dataset[sub+1] = [data, label]

        if transforms:
            trans = ComposeTransforms(transforms)
            trans.insert(0, SplitTrainTest(test_size, seed))
        else:
            trans = SplitTrainTest(test_size, seed)
        self._dataset = trans(dataset, verbose=verbose)