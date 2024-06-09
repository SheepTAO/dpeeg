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

from ..utils import loger, verbose, DPEEG_SEED, unpacked, get_init_args
from .utils import yield_data
from ..tools.logger import _Level
import dpeeg.data.functions as F
from .utils import total_trials
from .functions import (
    split_train_test,
    to_tensor,
    slide_win,
    save,
    cheby2_filter,
    label_mapping,
    pick_label,
)


class Transforms(abc.ABC):
    def __init__(self) -> None:
        self._repr = None
    
    @abc.abstractmethod
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        pass

    def __repr__(self) -> str:
        if self._repr:
            return self._repr
        else:
            class_name = self.__class__.__name__
            return f'{class_name} not implement attribute `self._repr`.'


class ComposeTransforms(Transforms):
    def __init__(self, *transforms : Transforms) -> None:
        '''Composes several transforms together. 

        The transforms in `ComposeTransforms` are connected in a cascading way.

        Parameters
        ----------
        transforms : sequential container of `Transforms`
            Sequential of transforms to compose. 

        Examples
        --------
        If you have multiple transforms that are processed sequentiallt, you 
        can do like:
        >>> from dpeeg.data import transforms
        >>> trans = transforms.ComposeTransforms(
        ...     transforms.Normalization(),
        ...     transforms.Unsqueeze(),
        ... )
        >>> trans
        ComposeTransforms(
            (0): Normalization(mode=z-score)
            (1): Unsqueeze(dim=1)
        )
        '''
        super().__init__()
        self.trans : list[Transforms] = []
        self.appends(*transforms)

    @verbose
    def __call__(self, input : dict, verbose : _Level = None):
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
        '''Append transforms to the last of composes.
        '''
        trans = unpacked(*transforms)
        for tran in trans:
            if isinstance(tran, ComposeTransforms):
                self.trans.extend(tran.get_data())
            else:
                self.trans.append(tran)
        
    def insert(self, index : int, transform : Transforms) -> None:
        '''Insert a transform at index.
        '''
        self.trans.insert(index, transform)

    def get_data(self) -> list[Transforms]:
        '''Return list of Transforms.
        '''
        return self.trans


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
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
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
    '''Convert the numpy data in the dataset into Tensor format.
    '''
    @verbose
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
        for sub in input.values():
            for mode in ['train', 'test']:
                sub[mode] = list(to_tensor(*sub[mode], verbose=verbose))
        return input

    def __repr__(self) -> str:
        return 'ToTensor'


class Normalization(Transforms):
    def __init__(
        self, 
        mode : str = 'z-score', 
        factor_new : float = 1e-3,
        verbose : _Level = None
    ) -> None:
        '''Normalize data in the given way in the given dimension.

        Parameters
        ----------
        mode : str
            within subject:
            - `z-score`,
                :math: $X_{i}=\\frac{X_{i}-mean(X_{i})}{std(X_{i})}$ where mean
                represents the channel-wise average value and std represents the 
                channel-wise standard deviation value, calculated with the training
                data and used directly for the test data.
            - `ems` (exponential moving standardize),
                Compute the exponental moving mean :math: $m_t$ at time $t$ as
                :math: $m_t=\\mathrm{factornew}\\cdot mean(x_t)+(1-\\mathrm{factornew})\\cdot m_{t-1}$.
                Then compute exponential moving variance $v_t$ at time t as
                :math: $v_t=\\mathrm{factornew}\\cdot (m_t-x_t)^2+(1-\\mathrm{factornew})\\cdot v_{t-1}$.
                Finally, standardize the data point :math: $x_t$ at time $t$ as:
                :math: $x'_t=(x_t-m_t)/max(\\sqrt{->v_t}, eps)$.
            cross subject:
            - `ea` (euclidean-space alignment),
                An unsupervised standardization of data transfer across subjects.
                > H. He and D. Wu, 'Transfer Learning for Brain-Computer Interfaces:
                A Euclidean Space Data Alignment Approach', Ieee T Bio-med Eng, vol. 
                67, no. 2, pp. 399-410, Feb. 2020, doi: 10.1109/TBME.2019.2913914.
            
        factor_new : float
            Smoothing factor of exponential moving standardize.

        Notes
        -----
        heavy development.
        '''
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.mode_list = ['z-score', 'ems', 'ea']
        self.mode = mode
        self.factor_new = factor_new

    @verbose
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        if self.mode not in self.mode_list:
            raise ValueError('Only the following normalization methods are '+
                             f'supported: {self.mode_list}')

        loger.info(f'[{self} starting] ...')
        R = np.empty(0)
        if self.mode == 'ea':
            dataList = []
            for sub in input.values():
                for mode in ['train', 'test']:
                    dataList.append(sub[mode][0] / np.max(np.abs(sub[mode][0])))
            data = np.concatenate(dataList)

            Rbar = np.zeros((data.shape[1], data.shape[1]))
            for i in range(len(data)):
                Rbar += np.dot(data[i], data[i].T)
            RbarMean = Rbar / len(data)
            R = np.linalg.inv(np.sqrt(RbarMean))

        for sub in input.values():

            if self.mode == 'z-score':
                m, s = np.mean(sub['train'][0]), np.std(sub['train'][0])
                for mode in ['train', 'test']:
                    sub[mode][0] = (sub[mode][0] - m) / s

            # BUG
            elif self.mode == 'ems':
                df = pd.DataFrame(sub['train'][0].T)
                meaned = df.ewm(alpha=self.factor_new).mean()
                for mode in ['train', 'test']:
                    df = pd.DataFrame(sub[mode][0].T)
                    demeaned = df - meaned
                    squared = demeaned * demeaned
                    squareEwmed = squared.ewm(alpha=self.factor_new).mean()
                    standardized = demeaned / np.sqrt(np.array(squareEwmed))
                    standardized = np.array(standardized)
                    sub[mode][0] = standardized.T

            # NOT TEST
            elif self.mode == 'ea':
                for mode in ['train', 'test']:
                    sub[mode][0] /= np.max(np.abs(sub[mode][0]))
                    sub[mode][0] = np.dot(sub[mode][0], R)
        return input


class SlideWin(Transforms):
    def __init__(self, win : int, overlap : int = 0) -> None:
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
        '''
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.win = win
        self.overlap = overlap

    @verbose
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input):
            sub_data[0], sub_data[1] = slide_win(
                sub_data[0], self.win, self.overlap, sub_data[1], verbose
            )
        return input


class Unsqueeze(Transforms):
    def __init__(self, dim : int = 1) -> None:
        '''Insert a dimension on the data.

        This transform is usually used to insert a empty dimension on EEG data.

        Parameters
        ----------
        dim : int
            Position in the expanded dim where the new dim is placed.
        '''
        super().__init__()
        self._repr = get_init_args(self, locals(), format='rp')
        self.dim = dim

    @verbose
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
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
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
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
            raise TypeError(f'Filter_bank must be a list, not {type(fb)}.')
        for f in fb:
            if len(f) != 2:
                raise ValueError(
                    'The filter should be of two variables low pass and high '
                    'pass cutoff frequency.'
                )
        return fb

    @verbose
    def __call__(self, input: dict, verbose: _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
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
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
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
    def __call__(self, input : dict, verbose : _Level = None) -> None:
        loger.info(f'[{self} starting] ...')
        save(self.folder, input, verbose=verbose)


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
    def __call__(self, input: dict, verbose: _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input):
            sub_data[1] = label_mapping(sub_data[1], self.mapping, self.order)
        return input


class PickLabel(Transforms):
    '''Pick a subset of data.

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
    def __call__(self, input: dict, verbose: _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
        for _, _, sub_data in yield_data(input):
            sub_data[0], sub_data[1] = pick_label(*sub_data, pick=self.pick)
        return input