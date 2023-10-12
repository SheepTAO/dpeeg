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
from typing import Optional, Callable, Union, List

from ..utils import loger, verbose, DPEEG_SEED, unpacked, DPEEG_LOGER_LEVEL
from ..tools.logger import _Level
from .functions import (
    split_train_test,
    to_tensor,
    slide_win,
    save,
)


class Transforms(abc.ABC):
    @abc.abstractmethod
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass


class ComposeTransforms(Transforms):
    '''Composes several transforms together.
    '''
    def __init__(
        self, 
        *transforms : Transforms, 
    ) -> None:
        '''Composes several transforms together. 

        The transforms in `ComposeTransforms` are connected in a cascading way.

        Parameters
        ----------
        transforms : sequential container of Transforms
            Transforms (sequential of `Transform` objects): sequential of tran-
            sforms to compose. 
        '''
        super().__init__()
        self.trans : List[Transforms] = []
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

    def get_data(self) -> List[Transforms]:
        '''Return list of Transforms.
        '''
        return self.trans


class SplitTrainTest(Transforms):
    '''Split the dataset into training and testing sets.
    '''
    def __init__(
        self, 
        testSize : float = .25, 
        seed : int = DPEEG_SEED, 
        sample : Optional[List[int]] = None,
    ) -> None:
        '''Split the dataset into training and testing sets.

        Parameters
        ----------
        testSize : float
            The proportion of the test set. Default is 0.25. If index is not 
            None, testSize will be ignored. Default use stratified fashion and 
            the last arr serves as the class labels.
        seed : int
            Random seed when splitting. Default is DPEEG_SEED.
        sample : list of int, optional
            A list of integers, the entries indicate which data were selected
            as the test set. If None, testSize will be used. Default is None.
        '''
        super().__init__()
        self.testSize = testSize
        self.seed = seed
        self.sample = sample

    @verbose
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
        for sub, data in input.items():
            trainX, testX, trainy, testy = split_train_test(
                data[0], data[1], testSize=self.testSize, seed=self.seed,
                sample=self.sample, verbose=verbose
            )
            input[sub] = {}
            input[sub]['train'] = [trainX, trainy]
            input[sub]['test'] = [testX, testy]
        return input

    def __repr__(self) -> str:
        s = 'SplitDataset('
        if self.sample:
            s += 'sample'
        else:
            s += f'testSize={self.testSize}'
            if self.seed:
                s += f', seed={self.seed}'
        return s + ')'


class ToTensor(Transforms):
    '''Convert the numpy data in the dataset into Tensor format.
    '''
    @verbose
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
        for sub in input.values():
            for mode in ['train', 'test']:
                sub[mode][0], sub[mode][1] = to_tensor(
                    sub[mode][0], sub[mode][1], verbose=verbose
                )
        return input

    def __repr__(self) -> str:
        return 'ToTensor'


class Normalization(Transforms):
    '''Normalize the data.
    '''
    def __init__(
        self, 
        mode : str = 'z-score', 
        factorNew : float = 1e-3,
        verbose : _Level = None
    ) -> None:
        '''Normalize data in the given way in the given dimension.

        Parameters
        ----------
        mode : str
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
            
        factorNew : float
            Smoothing factor of exponential moving standardize. Default is 1e-3.

        Notes
        -----
        heavy development
        '''
        super().__init__()
        self.modeList = ['z-score', 'ems', 'ea']
        self.mode = mode
        self.factorNew = factorNew

    @verbose
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        if self.mode not in self.modeList:
            raise ValueError('Only the following normalization methods are '+
                             f'supported: {self.modeList}')

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
                meaned = df.ewm(alpha=self.factorNew).mean()
                for mode in ['train', 'test']:
                    df = pd.DataFrame(sub[mode][0].T)
                    demeaned = df - meaned
                    squared = demeaned * demeaned
                    squareEwmed = squared.ewm(alpha=self.factorNew).mean()
                    standardized = demeaned / np.sqrt(np.array(squareEwmed))
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
        if self.mode == 'ems':
            s += f', factorNew={self.factorNew})'
        return s + ')'


class SlideWin(Transforms):
    '''Apply a sliding window to the dataset.
    '''
    def __init__(
        self, 
        win : int = 125, 
        overlap : int = 0,
    ) -> None:
        '''This transform is only splits the time series (dim = -1) through the
        sliding window operation on the original dataset. If the time axis is 
        not divisible by the sliding window, the last remaining time data will 
        be discarded.

        Parameters
        ----------
        win : int
            The size of the sliding window.
        overlap : int
            The amount of overlap between adjacent sliding windows. Default is 0.
        '''
        super().__init__()
        self.win = win
        self.overlap = overlap

    @verbose
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
        for sub in input.values():
            for mode in ['train', 'test']:
                sub[mode][0], sub[mode][1] = slide_win(
                    sub[mode][0], self.win, self.overlap,
                    sub[mode][1], verbose=verbose
                )
        return input

    def __repr__(self) -> str:
        s = f'SlideWin(win={self.win}'
        if self.overlap != 0:
            s += f', overlap={self.overlap}'
        return s + ')'


class Unsqueeze(Transforms):
    '''Insert a dimension on the data.
    '''
    def __init__(
        self,
        dim : int = 1,
    ) -> None:
        '''This function is usually used to insert a feature dimension on EEG data.

        Parameters
        ----------
        dim : int
            Position in the expanded dim where the new dim is placed.
        '''
        super().__init__()
        self.dim = dim

    @verbose
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
        for sub in input.values():
            for mode in ['train', 'test']:
                sub[mode][0] = np.expand_dims(sub[mode][0], self.dim)
        return input

    def __repr__(self) -> str:
        return f'Unsqueeze(dim={self.dim})'


class ApplyFunc(Transforms):
    '''Apply a function on data.
    '''
    def __init__(
        self, 
        func : Callable, 
    ) -> None:
        '''This transform can be used to change the data shape and so on.

        Examples
        --------
        If you want to pass a function with parameters, such as you want to
        use `np.expand_dims()` with `axis` parameter, you can do as follows:
        >>> from functools import partial
        >>> transforms.ApplyFunc(partial(np.expand_dims, axis=1))
        or
        >>> transforms.ApplyFunc(lambda x: np.expand_dims(x, 1))

        Notes
        -----
        heavy development
        '''
        super().__init__()
        self.func = func

    @verbose
    def __call__(self, input : dict, verbose : _Level = None) -> dict:
        loger.info(f'[{self} starting] ...')
        for sub in input.values():
            for mode in ['train', 'test']:
                sub[mode][0] = self.func(sub[mode][0])
        return input

    def __repr__(self) -> str:
        return f'ApplyFunc(func={self.func})'


class Save(Transforms):
    '''Save the transformed data.
    '''
    def __init__(
        self, 
        folder : str, 
    ) -> None:
        '''Save transformed dataset to a binary file in NumPy `.npy` format.

        Parameters
        ----------
        folder : str
            Folder name to save transformed data.
        '''
        super().__init__()
        self.folder = folder

    @verbose
    def __call__(self, input : dict, verbose : _Level = None) -> None:
        loger.info(f'[{self} starting] ...')
        save(self.folder, input, verbose=verbose)

    def __repr__(self) -> str:
        return f'Save(folder={self.folder})'
