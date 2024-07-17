#!/usr/bin/env python
# coding: utf-8

"""
    This module is used to perform common preprocessing on eeg data (ndarray). 
    All transforms is in-place.

    TODO: Data augmentation.
    
    @author: SheepTAO
    @data: 2023-5-15
"""


import abc


import numpy as np
import pandas as pd
from typing import Literal
from collections.abc import Callable
from numpy import ndarray
from mne.utils import logger, verbose


from dpeeg.data.base import EEGDataset
from ..utils import DPEEG_SEED, unpacked, get_init_args
from .utils import yield_data
from ..tools.logger import _Level
import dpeeg.data.functions as F
from .utils import total_trials
from .functions import (
    split_train_test,
    to_tensor,
    crop,
    z_score_norm,
    min_max_norm,
    slide_win,
    save,
    cheby2_filter,
    label_mapping,
    pick_label,
)


def yield_data(
    input : EEGDataset,
    mode : Literal['train', 'test', 'all'] = 'all'
):
    '''Traverse all subject training and test set in the dataset, and return 
    the current subject name, mode and data.
    '''
    for sub, sub_data in input.items():
        for m in ['train', 'test']:
            if mode != 'all' and m != mode:
                continue
            yield sub, m, sub_data[m]


class Transforms(abc.ABC):
    def __init__(self) -> None:
        self._repr = None
    
    @abc.abstractmethod
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        pass

    def __repr__(self) -> str:
        if self._repr:
            return self._repr
        else:
            class_name = self.__class__.__name__
            return f'{class_name} not implement attribute `self._repr`.'


class Sequential(Transforms):
    '''A sequential container.

    Transforms will be added to it in the order they are passed.

    Parameters
    ----------
    transforms : sequential of `Transforms`
        Sequential of transforms to compose. 

    Examples
    --------
    If you have multiple transforms that are processed sequentiallt, you can do
    like:
    >>> from dpeeg.data import transforms
    >>> trans = transforms.Sequential(
    ...     transforms.Unsqueeze(),
    ...     transforms.ToTensor(),
    ... )
    >>> trans
    ComposeTransforms(
        (0): Unsqueeze(dim=1)
        (1): ToTensor()
    )
    '''
    def __init__(self, *transforms : Transforms) -> None:
        super().__init__()
        self.trans : list[Transforms] = []
        self.appends(*transforms)

    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        for tran in self.trans:
            input = tran(input, verbose=verbose)
        return input

    def __repr__(self) -> str:
        s = 'ComposeTransforms('
        if len(self.trans) == 0:
            return s + ')'
        else:
            for idx, tran in enumerate(self.trans):
                s += f'\n ({idx}): {tran}'
        return s + '\n)'

    def appends(self, *transforms : Transforms) -> None:
        '''Append transforms to the last of composes.'''
        trans = unpacked(*transforms)
        for tran in trans:
            if isinstance(tran, Sequential):
                self.trans.extend(tran.get_trans())
            else:
                self.trans.append(tran)
        
    def insert(self, index : int, transform : Transforms) -> None:
        '''Insert a transform at index.'''
        self.trans.insert(index, transform)

    def get_trans(self) -> list[Transforms]:
        '''Return list of Transforms.'''
        return self.trans


class Identity(Transforms):
    '''Placeholder identity operator.'''
    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        return input
    
    def __repr__(self) -> str:
        return 'Identity()'
    


class SplitTrainTest(Transforms):
    def __init__(
        self, 
        test_size : float = .25, 
        seed : int = DPEEG_SEED, 
        sample : list[int] | None = None,
    ) -> None:
        '''Split the dataset into training and testing sets.

        Parameters
        ----------
        test_size : float
            The proportion of the test set. If index not None, test_size will 
            be ignored. Default use stratified fashion and the last arr serves 
            as the class label.
        seed : int
            Random seed when splitting.
        sample : list of int, optional
            A list of integers, the entries indicate which data were selected 
            as the test set. If None, test_size will be used.
        '''
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.test_size = test_size
        self.seed = seed
        self.sample = sample

    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        for sub, data in input.items():
            trainX, testX, trainy, testy = split_train_test(
                data[0], data[1], test_size=self.test_size, seed=self.seed,
                sample=self.sample, verbose=verbose
            )
            input[sub] = {}
            input[sub]['train'] = [trainX, trainy]
            input[sub]['test'] = [testX, testy]
        return input


class ToTensor(Transforms):
    '''Convert the numpy data in the dataset into Tensor format.'''
    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input):
            sub_data[0], sub_data[1] = to_tensor(*sub_data, verbose=verbose)
        return input

    def __repr__(self) -> str:
        return 'ToTensor()'


class Crop(Transforms):
    '''Crop a time interval.

    Parameters
    ----------
    tmin : int
        Start time of selection in sampling points.
    tmax : int
        End time of selection in sampling points.
    include_tmax : bool
        If `False`, exclude tmax.
    mode : str
        Transform only the training set, test set, or both.
    
    Return
    ------
    EEGDataset
        Transformed eegdataset.
    '''
    def __init__(
        self,
        tmin : int | None = None,
        tmax : int | None = None,
        include_tmax : bool = False,
        mode : Literal['train', 'test', 'all'] = 'all'
    ) -> None:
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.tmin = tmin
        self.tmax = tmax
        self.include_tmax = include_tmax
        self.mode = mode

    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input, self.mode): # type: ignore
            sub_data[0] = crop(
                sub_data[0], self.tmin, self.tmax, self.include_tmax, verbose
            )
        return input


class ZscoreNorm(Transforms):
    '''Z-score normalization per subject.

    Parameters
    ----------
    dim : int, optional
        The dimension to normalize. Usually, -1 for channels and -2 for time
        points. If None, normalize at the sample level.

    Return
    ------
    EEGDataset
        Transformed eegdataset.
    '''
    def __init__(self, dim : int | None = None) -> None:
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.dim = dim

    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input):
            sub_data[0] = z_score_norm(sub_data[0], self.dim, verbose=verbose)
        return input


class MinMaxNorm(Transforms):
    '''Min-max normalization per subject.


    Parameters
    ----------
    dim : int, optional
        The dimension to normalize. Usually, -1 for channels and -2 for time
        points. If None, normalize at the sample level.

    Return
    ------
    EEGDataset
        Transformed eegdataset.
    '''
    def __init__(self, dim : int | None = None) -> None:
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.dim = dim

    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input):
            sub_data[0] = min_max_norm(sub_data[0], self.dim, verbose=verbose)
        return input


class SlideWin(Transforms):
    def __init__(
        self, 
        win : int, 
        overlap : int = 0,
        mode : Literal['train', 'test', 'all'] = 'all'
    ) -> None:
        '''Apply a sliding window to the dataset.

        This transform is only splits the time series (dim = -1) through the 
        sliding window operation on the original dataset. If the time axis is 
        not divisible by the sliding window, the last remaining time data will 
        be discarded.

        Parameters
        ----------
        win : int
            The size of the sliding window.
        overlap : int
            The amount of overlap between adjacent sliding windows.
        mode : str
            Transform only the training set, test set, or both.
        '''
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.win = win
        self.overlap = overlap
        self.mode = mode

    @verbose
    def __call__(self, input : EEGDataset, verbose : _Level = 'WARNING') -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input, self.mode): # type: ignore
            sub_data[0], sub_data[1] = slide_win(
                sub_data[0], self.win, self.overlap, sub_data[1], verbose
            )
        return input


class Unsqueeze(Transforms):
    '''Insert a dimension on the data.

    This transform is usually used to insert a empty dimension on EEG data.

    Parameters
    ----------
    dim : int
        Position in the expanded dim where the new dim is placed.

    Return
    ------
    dict
        Transformed dataset.
    '''
    def __init__(self, dim : int = 1) -> None:
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.dim = dim

    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input):
            sub_data[0] = np.expand_dims(sub_data[0], self.dim)
        return input


class Augmentation(Transforms):
    def __init__(
        self,
        method : str,
        only_train : bool = True,
        **kwargs
    ) -> None:
        '''Training set data augmentation.

        This transform is mainly used for data augmentation of the data set.

        Parameters
        ----------
        method : str
            Specified data augmentation method.
        only_train : bool
            If True, data augmentation is performed only on the training set.
        kwargs : dict
            Parameters of the corresponding augmentation method.

        See Also
        --------
        dpeeg.data.functions : 
            Get detailed data augmentation method parameters.
        '''
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.aug = getattr(F, method)
        self.mode = 'train' if only_train else 'all'
        self.kwargs = kwargs

    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input, self.mode): # type: ignore
            sub_data[0], sub_data[1] = self.aug(*sub_data, **self.kwargs)
        return input


class FilterBank(Transforms):
    def __init__(
        self,
        freq : float,
        filter_bank : list,
        transition_bandwidth : float = 2.,
        gstop : float = 30,
        gpass : float = 3,
    ) -> None:
        '''Filter Bank.

        EEG data will be filtered according to different filtering frequencies 
        and finally concatenated together. eg.`(Batch, ...) -> (Batch, F, ...)`
        if the number of filter banks exceeds 1, `(Batch, ...) -> (Batch, ...)`
        if the filter has only one.

        Parameters
        ----------
        freq : float
            Data sampling frequency.
        filter_bank : multiple 2 float of list
            The low-pass and high-pass cutoff frequencies for each filter set.
        transition_bandwidth : float
            The bandwidth (in hertz) of the transition region of the frequency 
            response from the passband to the stopband.
        gstop : float
            The minimum attenuation in the stopband (dB).
        gpass : float
            The maximum loss in the passband (dB).
        '''
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.freq = freq
        self.filter_bank = self._check_filter_bank(filter_bank)
        self.transition_bandwidth = transition_bandwidth
        self.gpass = gpass
        self.gstop = gstop

    def _check_filter_bank(self, fb):
        if not isinstance(fb, list):
            raise TypeError(f'filter_bank must be a list, not {type(fb)}.')
        for f in fb:
            if len(f) != 2:
                raise ValueError(
                    'The filter should be of two variables low pass and high '
                    'pass cutoff frequency.'
                )
        return fb

    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        bank_len = len(self.filter_bank)
        for sub, mode, sub_data in yield_data(input):
            trials = total_trials(input, sub, mode) # type: ignore
            data = np.empty((trials, bank_len, *sub_data[0].shape[1:]))

            for i, cutoff in enumerate(self.filter_bank):
                filter_data = cheby2_filter(
                    data=sub_data[0],
                    freq=self.freq,
                    l_freq=cutoff[0],
                    h_freq=cutoff[1],
                    transition_bandwidth=self.transition_bandwidth,
                    gpass=self.gpass,
                    gstop=self.gstop,
                    verbose=verbose,
                )
                data[:, i] = filter_data

            if bank_len == 1:
                data = np.squeeze(data, 1)
            sub_data[0] = data

        return input


class ApplyFunc(Transforms):
    def __init__(
        self, 
        func : Callable[..., ndarray],
        mode : Literal['train', 'test', 'all'] = 'all',
        data : bool = True,
        **kwargs
    ) -> None:
        '''Apply a custom function to training or test data or label.

        Parameters
        ----------
        func : Callable
            Transformation data callback function. The first parameter of the
            function must be ndarray data.
        mode : str
            Transform only the training set, test set, or both.
        data : bool
            If True, apply to the data. Otherwise, apply to the label.
        kwargs : dict
            Keyword arguments for callback function.

        Examples
        --------
        If you want to pass a function with parameters, such as you want to use 
        `np.expand_dims()` with `axis` parameter, you can do as follows:
        >>> from functools import partial
        >>> transforms.ApplyFunc(partial(np.expand_dims, axis=1))

        or

        >>> transforms.ApplyFunc(lambda x: np.expand_dims(x, 1))
        '''
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.func = func
        self.mode = mode
        self.data = data
        self.kwargs = kwargs

    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input, self.mode): # type: ignore
            if self.data:
                sub_data[0] = self.func(sub_data[0], **self.kwargs)
            else:
                sub_data[1] = self.func(sub_data[1], **self.kwargs)
        return input


class Save(Transforms):
    def __init__(self, folder : str) -> None:
        '''Save the transformed data.

        Save transformed dataset to a binary file in NumPy `.npy` format.

        Parameters
        ----------
        folder : str
            Folder name to save transformed data.
        '''
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.folder = folder

    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        save(self.folder, input, verbose=verbose)
        return input


class LabelMapping(Transforms):
    '''Rearrange the original label according to mapping rules.

    Parameters
    ----------
    mapping : ndarray (2, label_num)
        Label mapping relationship.
    order : bool
        New label must start from 0.
    '''
    def __init__(self, mapping : ndarray, order : bool = True) -> None:
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.mapping = mapping
        self.order = order

    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input):
            sub_data[1] = label_mapping(sub_data[1], self.mapping, self.order)
        return input


class PickLabel(Transforms):
    '''Pick a subset of data.

    Pick the required labels and data from the dataset and re-label them.

    Parameters
    ----------
    pick : ndarray
        Label to include.
    '''
    def __init__(self, pick : ndarray) -> None:
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.pick = pick

    @verbose
    def __call__(self, input : EEGDataset, verbose = None) -> EEGDataset:
        logger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input):
            sub_data[0], sub_data[1] = pick_label(*sub_data, pick=self.pick)
        return input