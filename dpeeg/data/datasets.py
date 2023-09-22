#!/usr/bin/env python
# coding: utf-8

"""
    This module is used to load commonly used MI-EEG data.
    
    NOTE: All data is read during the initialization phase.
        - Defect: At the beginning, all the data needs to be read once,
        and the initialization time is longer. If the amount of data is 
        huge, the memory consumption is huge.
        - Advantage: In the process of multiple iterations, there is no
        need to read data repeatedly, and only one round of IO operation
        is required, which saves time for multiple rounds of training.
        
    NOTE: If you want to build your own data set, you can refer to the
        sample dataset to build it. Please reassign the `_eventId` and
        `_raw` attributes. In the final stage of initialization, please
        call the `self._load_data()` to completely initialize your dataset.
        
    
    @Author  : SheepTAO
    @Time    : 2023-07-26
"""

import mne
from typing import Optional, List, Union

from .transforms import Transforms, ComposeTransforms, SplitTrainTest
from ..utils import loger, verbose, DPEEG_SEED


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
        return tuple(sub['train'][0].shape)
        
    @property
    def eventId(self) -> dict:
        '''Return the event label of the dataset and its corresponding id.'''
        return self._check_attr('_eventId')
    
    @property
    def classes(self) -> tuple:
        '''Return the event label of the dataset.'''
        return tuple(self.eventId.keys())
    
    @property
    def raw(self) -> dict:
        '''Return the original Epochs data.'''
        return self._check_attr('_raw')
    
    @property
    def dataset(self) -> dict:
        '''Return the loaded data.'''
        return self._check_attr('_dataset')

    def __init__(
        self,
        transforms : Optional[Union[List[Transforms], Transforms]] = None,
        testSize : float = .2, 
        seed : int = DPEEG_SEED,
        verbose : Optional[str] = None
    ) -> None:
        '''EEG Dataset abstract base class.

        Parameters
        ----------
        %(subjects)s : list of int, optional
            List of subject number. If None, all subjects will be loaded. Default is None.
        %(tmin, tmax)s : float
            Start and end time of the epochs in seconds, relative to the time-locked event.
            The closest or matching samples corresponding to the start and end time are 
            included. Default is start and end time of epochs, respectively.
        %(picks)s : list of str, optional
            Channels to include. If None, will pick all channels. Default is None.
        transforms : list of Callable, optional
            List of pre-transforms on dataset. Default is None.
        testSize : float
            Split the training set and test set proportions. If the dataset is already split,
            it will be ignored. Default is 0.2.
        seed : int
            Random seed when splitting. Default is DPEEG_SEED.
        verbose : str, optional
            Log level of mne. Default is None.
        '''
        super().__init__()

        mne.set_log_level(verbose)
        
        self._testSize = testSize
        self._seed = seed
        self._transforms = transforms
        self._dataset = None
        
        # NOTE
        # Please make sure the following attributes are correctly overridden
        self._eventId = None        # task name and its corresponding label
        self._raw = None            # each subject and its corresponding Epochs
    
    def _load_data(self, split : bool = False) -> None:
        '''Extract data from Epochs and split.
        
        split : bool, optional
            Whether `self.raw` has been splited. Default is False.
        NOTE: Avoid data leakage when you split data.
        '''
        loger.info('Loading data from Epochs ...')
        dataset = {}
        for sub, sEpochs in self.raw.items():
            if not split:
                data = sEpochs.get_data()[:, :, :-1]
                label = sEpochs.events[:, -1]
                dataset[sub] = [data, label]
            else:
                dataset[sub] = {}
                for mode, mEpochs in sEpochs.items():
                    data = mEpochs.get_data()[:, :, :-1]
                    label = mEpochs.events[:, -1]
                    dataset[sub][mode] = [data, label]

        # split the dataset before transforms
        if self._transforms:
            trans = ComposeTransforms(self._transforms)
            if not split:
                trans.insert(0, SplitTrainTest(self._testSize, self._seed))
            self._dataset = trans(dataset)
        else:
            self._dataset = dataset
        loger.info('Loading dataset done.')
        
    
    def __getitem__(self, args) -> dict:
        '''Return the index position data of the corresponding subject
        and the corresponding set.
        
        Parameters
        ----------
        args : sub, index
        
        sub : int
            Number of subject.
        index : int, slice, optional
            The data index corresponding to the training set. If index is None,
            all the data (train and test) of the subject will be returned as a
            dictionary. Default is None.
        
        Returns
        -------
        dict - A dict contain train set (and test set).

        Example
        -------
            >>> dataset[1]
            {'train': (sub1[data], sub1[label]), 
             'test': (sub1[data], sub1[label])}
            >>> dataset[2, :2]
            (sub2[data[:2]], sub2[label[:2]])
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
        
        if index:
            return {
                'train' : (self.dataset[sub]['train'][0][index],
                           self.dataset[sub]['train'][1][index])
            }
        else:
            return {
                'train': tuple(self.dataset[sub]['train']),
                'test': tuple(self.dataset[sub]['test'])
            }
        
    def __len__(self) -> int:
        return len(self.raw)
    
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
        subjects : Optional[List[int]] = None,
        tmin : float = 0,
        tmax : float = 1,
        transforms : Optional[Union[List[Transforms], Transforms]] = None,
        testSize : float = .2,
        picks : Optional[List[str]] = None,
        baseline = None,
        seed : int = DPEEG_SEED,
        verbose : Optional[str] = None,
        **epoArgs
    ) -> None:
        '''Physionet MI Dataset.
        '''
        super().__init__(transforms, testSize, seed, verbose)
        loger.info('Reading PhysionetMI Dataset ...')
        
        from moabb.datasets import PhysionetMI
        self._baseRaw = PhysionetMI().get_data(subjects)
        self._badSub = [88, 92, 100, 104]
        self._eventId = {
            'left_hand': 0,
            'right_hand': 1,
            'hands': 2,
            'feet': 3
        }
        
        self._raw = {}
        for sub, sess in self._baseRaw.copy().items():
            loger.info(f'\nCreate sub{sub} Epochs ...')
            if sub in self._badSub:
                continue
            
            epochsSes = []
            for run in sess['session_0'].values():
                
                ann = mne.events_from_annotations(run, regexp='^(?!rest$).*$')
                # update events (labels)
                ann[0][:, -1] = ann[0][:, -1]-1 if 'left_hand' in ann[1].keys() \
                    else ann[0][:, -1]+1
                # update event_id
                for key in ann[1].keys():
                    ann[1][key] = self._eventId[key]
                
                events = ann[0][:-1]
                epochsSes.append(
                    mne.Epochs(run, events, ann[1], tmin, tmax, baseline, picks,
                               preload=True, **epoArgs)
                )
            
            self._raw[sub] = mne.concatenate_epochs(epochsSes)
        
        self._load_data()
            

class BCICIV2A(EEGDataset):
    '''BCI Competition IV Data sets 2a.
    channels=22; subjects=1-9; tasks={0:left hand, 1:right hand, 2:feet, 3:tongue};
    duration=4s; freq=250Hz; sessions=2.
    '''
    @verbose
    def __init__(
        self,
        subjects : Optional[List[int]] = None,
        tmin : float = 0,
        tmax : float = 4,
        transforms : Optional[Union[List[Transforms], Transforms]] = None,
        testSize : float = .2,
        mode : int = 1,
        picks : Optional[List[str]] = None,
        baseline = None,
        seed : int = DPEEG_SEED,
        verbose : Optional[str] = None,
        **epoArgs
    ) -> None:
        '''BCI Competition IV Data sets 2a.

        Parameters
        ----------
        mode: int, optional
            If mode = 0, training data and test data will only use session 1.
            If mode = 1, training data and test data will use session 1 and 2, respectively.
            If mode = 2, training data and test data will use both session 1 and 2.
            Default is 1.
        '''
        super().__init__(transforms, testSize, seed, verbose)
        loger.info('Reading BCICIV 2A Dataset ...')

        from moabb.datasets import BNCI2014001
        self._baseRaw = BNCI2014001().get_data(subjects)
        self._eventId = {
            'left_hand': 0,
            'right_hand': 1,
            'feet': 2,
            'tongue': 3
        }
        
        self._raw = {}
        for sub, data1 in self._baseRaw.copy().items():
            loger.info(f'Create sub{sub} Epochs ...')
            
            epochsSesOne, epochsSesTwo = [], []
            for session, data2 in data1.items():
                
                for run in data2.values():
                    
                    events = mne.find_events(run, 'stim')
                    events[:, -1] -= 1
                    epochs = mne.Epochs(run, events, self._eventId, tmin, tmax,
                                        baseline, picks, preload=True, **epoArgs)
                    epochs.drop_channels(['stim', 'EOG1', 'EOG2', 'EOG3'])
                    
                    if session == 'session_T':
                        epochsSesOne.append(epochs)
                    else:
                        epochsSesTwo.append(epochs)
            
            if mode == 0:
                self._raw[sub] = mne.concatenate_epochs(epochsSesOne)
            elif mode == 1:
                self._raw[sub] = {}
                self._raw[sub]['train'] = mne.concatenate_epochs(epochsSesOne)
                self._raw[sub]['test'] = mne.concatenate_epochs(epochsSesTwo)
            elif mode == 2:
                self._raw[sub] = mne.concatenate_epochs(epochsSesOne + epochsSesTwo)
            else:
                raise ValueError(f'Mode can only be 0, 1 and 2, but got {mode}.')
        
        split = True if mode == 1 else False
        self._load_data(split)


class HGD(EEGDataset):
    '''High Gamma Dataset.
    channels=128; subjects=1-14; tasks={0:feet, 1:left hand, 2:rest, 3:right hand};
    duration=4s; freq=500Hz; sessions=1.
    '''
    @verbose
    def __init__(
      self,
      subjects : Optional[List[int]] = None,
      tmin : float = 0,
      tmax : float = 4,
      transforms : Optional[Union[List[Transforms], Transforms]] = None,
      testSize : float = .2,
      picks : Optional[List[str]] = None,
      baseline = None,
      seed : int = DPEEG_SEED,
      verbose : Optional[str] = None,
      **epoArgs,
    ) -> None:
        '''High Gamma Dataset.
        '''
        super().__init__(transforms, testSize, seed, verbose)
        loger.info('Reading High Gamma Dataset ...')

        from moabb.datasets import Schirrmeister2017
        self._baseRaw = Schirrmeister2017().get_data(subjects)
        self._eventId = {
            'feet': 0, 
            'left_hand': 1, 
            'rest': 2, 
            'right_hand': 3
        }

        self._raw = {}
        for sub, sess in self._baseRaw.items():
            loger.info(f'Create sub{sub} Epochs ...')
            for mode, run in sess['session_0'].items():
                events, _ = mne.events_from_annotations(run)
                # update events
                events[:, -1] -= 1
                epochs = mne.Epochs(run, events, self._eventId, tmin, tmax,
                                    baseline, picks, preload=True, **epoArgs)
                self._raw.setdefault(sub, {})[mode] = epochs

        self._load_data(split=True)
