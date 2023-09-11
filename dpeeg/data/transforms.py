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
from typing import Any, Optional, Callable


class Compose:
    '''Composes several transforms together.
    '''
    def __init__(self, transforms : list) -> None:
        '''
        transforms : list
            Transforms (list of `Transform` objects): list of transforms to compose.
        '''
        self.transforms = transforms
        
    def __call__(self, input):
        for t in self.transforms:
            input = t(input)
        
        return input

    def __repr__(self) -> str:
        names = [trans.__class__.__name__ for trans in self.transforms]
        names = '\n'.join(names)
        
        return names

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
    def __init__(self, testSize : float = .2, seed : Optional[int] = None) -> None:
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
        
        for sub, data in input.items():
            trainX, testX, trainy, testy = \
                train_test_split(data[0], data[1], test_size=self.testSize,
                                 random_state=self.seed, stratify=data[1])

            input[sub] = {}
            input[sub]['train'] = [trainX, trainy]
            input[sub]['test'] = [testX, testy]

        return input


class ToTensor:
    '''Convert the numpy data in the dataset into Tensor format.
    '''
    def __init__(self) -> None:
        pass

    def __call__(self, input : dict) -> dict:
        for sub in input.values():
            for mode in ['train', 'test']:
                sub[mode][0] = torch.as_tensor(sub[mode][0]).float()
                sub[mode][1] = torch.as_tensor(sub[mode][1]).long()

        return input


class Normalization:
    '''Normalize the data.
    '''
    def __init__(
        self, mode : str = 'z-score', 
        factorNew : float = 1e-3,
        eps : float = 1e-4
    ) -> None:
        '''Normalize data in the given way in the given dimension.
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


class SlideWin:
    '''Apply a sliding window to the dataset.
    '''
    def __init__(self, win : int = 125, overlap : int = 0) -> None:
        '''This transform is only splits the time series(dim = -1) through the sliding 
        window operation on the original dataset. If the time axis is not divisible by 
        the sliding window, the last remaining time data will be discarded.

        win : int, optional
            The size of the sliding window. Default is 125.
        overlap : int, optional
            The amount of overlap between adjacent sliding windows. Default is 0.
        '''
        self.win = win
        self.overlap = overlap
        assert self.overlap < self.win, \
            f'overlap({overlap}) should be smaller than win({win}).'

    def __call__(self, input : dict) -> dict:
        assert self.overlap < self.win
        
        winSize = self.win - self.overlap
        for sub in input.values():
            for mode in ['train', 'test']:
                
                dataList = []
                sldNum = sub[mode][0].shape[-1] // winSize
                for i in range(sldNum):
                    if i == 0:
                        sld = sub[mode][0][..., i*winSize : (i+1)*winSize]
                    else:
                        sld = sub[mode][0][..., i*winSize-self.overlap : (i+1)*winSize]
                    dataList.append(sld)
                
                sub[mode][0] = np.concatenate(dataList)
                sub[mode][1] = np.repeat(sub[mode][1], sldNum)
        
        return input


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

class Save:
    '''Save the transformed data.
    '''
    def __init__(self, fileName : str, overwrite = False) -> None:
        '''Save transformed dataset to a binary file in NumPy `.npy` format.

        Parameters
        ----------
        fileName : str
            File or filename to which the data is saved, and `a .npy` extension will
            be appended to the filename if it does not already have one.
        overwrite : bool, optional
            If True, overwrite the destination file if it exists. Default is False.
        '''
        self.fileName = os.path.abspath(fileName)
        if self.fileName.endswith('.npy'):
            self.fileName += '.npy'
        if not overwrite and os.path.exists(self.fileName):
            raise FileExistsError('Data overwrite is not allowed.')

    def __call__(self, input : dict) -> Any:
        np.save(self.fileName, input)