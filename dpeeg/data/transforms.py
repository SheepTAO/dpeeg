#!/usr/bin/env python
# coding: utf-8

"""
    This module is used to perform common preprocessing on eeg data.

    TODO: Data augmentation.
    
    @author: SheepTAO
    @data: 2023-5-15
"""


import os
import torch
import numpy as np
import pandas as pd
from ..tools.logger import loger, verbose
from .functions import (
    slide_win,
    save,
)
from typing import Any, Optional, Callable, Union


class Compose:
    '''Composes several transforms together.
    '''
    @verbose
    def __init__(
        self, 
        transforms : list, 
        verbose : Optional[Union[int, str]] = None,
    ) -> None:
        '''
        transforms : list
            Transforms (list of `Transform` objects): list of transforms to compose.
        verbose : int, str, optional
            The log level of the entire transformation list. Default is None (INFO).
        '''
        self.transforms = transforms
        
    def __call__(self, input):
        loger.info('Transform dataset ...')
        loger.info('---------------------')
        for t in self.transforms:
            input = t(input)
        loger.info('---------------')
        loger.info('Transform done.')
        return input

    def __repr__(self) -> str:
        s = 'Compose('
        if len(self.transforms) == 0:
            return s + ')'
        else:
            for idx, tran in enumerate(self.transforms):
                s += f'\n ({idx}): {tran}'
        return s + '\n)'

    def appends(self, transforms):
        '''Append a transform or a list of transforms to the last of composes.
        '''
        if isinstance(transforms, list):
            self.transforms.extend(transforms)
        else:
            self.transforms.append(transforms)
        
    def insert(self, index, transform):
        '''Insert a transform at index.'''
        self.transforms.insert(index, transform)
    

class SplitDataset:
    '''Split the dataset into training and testing sets.
    '''
    @verbose
    def __init__(
        self, 
        testSize : float = .2, 
        seed : Optional[int] = None,
        verbose = None
    ) -> None:
        '''
        testSize : float, optional
            The proportion of the test set. Default is 0.2.
        seed : int, optional
            Random seed when splitting. Default is None.
        '''
        self.testSize = testSize
        self.seed = seed

    def __call__(self, input : dict) -> dict:
        
        from sklearn.model_selection import train_test_split

        loger.info(f'{str(self)} ...')
        for sub, data in input.items():
            trainX, testX, trainy, testy = \
                train_test_split(data[0], data[1], test_size=self.testSize,
                                 random_state=self.seed, stratify=data[1])

            input[sub] = {}
            input[sub]['train'] = [trainX, trainy]
            input[sub]['test'] = [testX, testy]
        return input

    def __repr__(self) -> str:
        s = f'SplitDataset(testSize={self.testSize}'
        if self.seed:
            s += f', seed={self.seed}'
        return s + ')'


class ToTensor:
    '''Convert the numpy data in the dataset into Tensor format.
    '''
    def __init__(self, verbose) -> None:
        pass

    def __call__(self, input : dict) -> dict:
        loger.info('Convert data to tensor format ...')
        for sub in input.values():
            for mode in ['train', 'test']:
                sub[mode][0] = torch.as_tensor(sub[mode][0]).float()
                sub[mode][1] = torch.as_tensor(sub[mode][1]).long()
        return input

    def __repr__(self) -> str:
        return 'ToTensor'


class Normalization:
    '''Normalize the data.
    '''
    @verbose
    def __init__(
        self, 
        mode : str = 'z-score', 
        factorNew : float = 1e-3,
        eps : float = 1e-4,
        verbose = None
    ) -> None:
        '''Normalize data in the given way in the given dimension.

        Parameters
        ----------
        mode : str, optional
            within subject:
            - `z-score`,
                :math: $X_{i}=\\frac{X_{i}-mean(X_{i})}{std(X_{i})}$ where mean
                represents the channel-wise average value and std represents
                the channel-wise standard deviation value, calculated with the 
                training data and used directly for the test data.
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
                > H. He and D. Wu, 'Transfer Learning for Brain-Computer Interfaces: A Euclidean Space 
                Data Alignment Approach', Ieee T Bio-med Eng, vol. 67, no. 2, pp. 399-410, Feb. 2020, 
                doi: 10.1109/TBME.2019.2913914.

            Default is z-score.
            
        factorNew : float, optional
            Smoothing factor of exponential moving standardize. Default is 1e-3.
        eps : float, optional
            Stabilizer for division by zero variance. Default is 1e-4.
        '''
        self.modeList = ['z-score', 'ems', 'ea']
        self.mode = mode
        self.factorNew = factorNew
        self.eps = eps

    def __call__(self, input : dict) -> dict:
        if self.mode not in self.modeList:
            raise ValueError('Only the following normalization methods are '+
                             f'supported: {self.modeList}')

        loger.info(f'{self} starting ...')
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
                    sub[mode][0] = (sub[mode][0] - m) / np.maximum(self.eps, s)

            # BUG
            elif self.mode == 'ems':
                df = pd.DataFrame(sub['train'][0].T)
                meaned = df.ewm(alpha=self.factorNew).mean()
                for mode in ['train', 'test']:
                    df = pd.DataFrame(sub[mode][0].T)
                    demeaned = df - meaned
                    squared = demeaned * demeaned
                    squareEwmed = squared.ewm(alpha=self.factorNew).mean()
                    standardized = demeaned / np.maximum(self.eps, 
                                            np.sqrt(np.array(squareEwmed)))
                    standardized = np.array(standardized)
                    sub[mode][0] = standardized.T

            # NOT TEST
            elif self.mode == 'ea':
                for mode in ['train', 'test']:
                    sub[mode][0] /= np.max(np.abs(sub[mode][0]))
                    sub[mode][0] = np.dot(sub[mode][0], R)
        return input
    
    def __repr__(self) -> str:
        s = f'Normalization(mode={self.mode}'
        if self.mode == 'z-score':
            return s + ')'
        elif self.mode == 'ems':
            return s + f', factorNew={self.factorNew}, eps={self.eps})'


class SlideWin:
    '''Apply a sliding window to the dataset.
    '''
    @verbose
    def __init__(
        self, 
        win : int = 125, 
        overlap : int = 0,
        verbose = None
    ) -> None:
        self.win = win
        self.overlap = overlap
        self.verbose = verbose

    def __call__(self, input : dict) -> dict:
        loger.info(f'{self} starting ...')
        for sub, data in input.items():
            loger.info(f'Sliding window to sub_{sub}.')
            for mode in ['train', 'test']:
                data[mode][0], data[mode][1] = slide_win(
                    data[mode][0], self.win, self.overlap,
                    data[mode][1], verbose='WARNING'
                )
        return input
    
    def __repr__(self) -> str:
        s = f'SlideWin(win={self.win}'
        if self.overlap != 0:
            s += f', overlap={self.overlap}'
        return s + ')'


class ApplyFunc:
    '''Apply a function on data.
    '''
    def __init__(self, func : Callable) -> None:
        '''This transform can be used to filter the data (via the `mne` library or
        other methods), change the data shape (via the `numpy`) and so on.

        NOTE:
        If you want to pass a function with parameters, such as you want to
        use `np.expand_dims()` with `axis` parameter, you can do as follows:
        >>> from functools import partial
        >>> transforms.ApplyFunc(partial(np.expand_dims, axis=1))
        or
        >>> transforms.ApplyFunc(lambda x: np.expand_dims(x, 1))
        '''
        self.func = func

    def __call__(self, input : dict) -> dict:
        for sub in input.values():
            for mode in ['train', 'test']:
                sub[mode][0] = self.func(sub[mode][0])
        return input
    
    def __repr__(self) -> str:
        return f'ApplyFunc(func={self.func})'


class Save:
    '''Save the transformed data.
    '''
    @verbose
    def __init__(
        self, 
        folder : str, 
        verbose = None
    ) -> None:
        self.folder = folder
        self.verbose = verbose

    def __call__(self, input : dict) -> None:
        save(self.folder, input, verbose=self.verbose)
    
    def __repr__(self) -> str:
        return f'Save(folder={self.folder})'
