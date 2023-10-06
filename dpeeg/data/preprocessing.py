#!/usr/bin/env python
# coding: utf-8

"""
    Mainly apply corresponding mne library functions to implement preprocessing
    before extracting the data for Epochs data.

    @Author  : SheepTAO
    @Time    : 2023-10-05
"""


import abc
from typing import Optional, Union, List


def dict_to_str(kwargs : dict, symbol : str = ', ') -> str:
    '''Convert the dictionary into a string format.

    Parameters
    ----------
    kwargs : dict
        The dictionary to be converted.
    symbol : str
        Join all key-value pairs with the specified separator character.
        Default is ', '.
    '''
    s = [f'{k}={v}' for k, v in kwargs.items()]
    return symbol.join(s)


class Preprocess(abc.ABC):
    @abc.abstractmethod
    def __call__(self, input : dict) -> dict:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass


class ComposePreprocess(Preprocess):
    def __init__(
        self, 
        preprocess : Union[List[Preprocess], Preprocess], 
    ) -> None:
        '''Compose serval preprocess together.

        The preprocess in `ComposePreprocess` are connected in a cascading way.

        Parameters
        ----------
        preprocess : list of Preprocess, Preprocess
            Preprocess (list of `Preprocess` objects): list of preprocess to 
            compose. If is Preprocess, then will be turned into a list contain-
            ing input.
        '''
        self.pres = []
        if isinstance(preprocess, Preprocess):
            if isinstance(preprocess, ComposePreprocess):
                self.pres.extend(preprocess.get_data())
            else:
                self.pres.append(preprocess)
        else:
            self.pres = preprocess

    def __call__(self, input: dict) -> dict:
        for t in self.pres:
            input = t(input)
        return input

    def __repr__(self) -> str:
        s = 'ComposePreprocess('
        if len(self.pres) == 0:
            return s + ')'
        else:
            for idx, pre in enumerate(self.pres):
                s += f'\n ({idx}): {pre}'
        return s + '\n)'

    def appends(self, preprocess : Union[List[Preprocess], Preprocess]) -> None:
        '''Append a preprocess or a list of preprocess to the last of composes.
        '''
        if isinstance(preprocess, list):
            self.pres.extend(preprocess)
        else:
            self.pres.append(preprocess)

    def insert(self, index : int, preprocess : Preprocess) -> None:
        '''Insert a preprocess at index.
        '''
        self.pres.insert(index, preprocess)

    def get_data(self) -> List[Preprocess]:
        '''Return list of Preprocess.
        '''
        return self.pres


class Filter(Preprocess):
    def __init__(
        self, 
        lfreq : Optional[float] = None,
        hfreq : Optional[float] = None,
        **mneFilterKwargs,
    ) -> None:
        '''Applies filter to the signals in epochs. The epoch will be modified 
        in-place.

        Parameters
        ----------
        lfreq : float, optional
            For FIR filters, the lower pass-band edge; for IIR filters, the 
            lower cutoff frequency. If None the data are only low-passed.
        hfreq : float, optional
            For FIR filters, the upper pass-band edge; for IIR filters, the up-
            per cutoff frequency. If None the data are only high-passed.
        mneFilterKwargs : dict
            Keyword arguments for filtering supported by `mne.io.Epochs.filter`
            Please refer to mne for a detailed explanation.
        '''
        self.lfreq = lfreq
        self.hfreq = hfreq
        self.mneFilterKwargs = mneFilterKwargs

    def __call__(self, input: dict) -> dict:
        for sub in input.values():
            if isinstance(sub, dict):
                for epoch in sub.values():
                    epoch.filter(self.lfreq, self.hfreq, **self.mneFilterKwargs)
            else:
                sub.filter(self.lfreq, self.hfreq, **self.mneFilterKwargs)
        return input

    def __repr__(self) -> str:
        s = f'Filter(lFreq={self.lfreq}, hFreq={self.hfreq}'
        if self.mneFilterKwargs:
            s += f', {dict_to_str(self.mneFilterKwargs)}'
        return s + ')'


class Resample(Preprocess):
    def __init__(
        self,
        sfreq : float,
        **mneResampleKwargs 
    ) -> None:
        '''Resample data in epochs. The epoch will be modified in-place.

        Parameters
        ----------
        sfreq : float
            New sample rate to use.
        mneResampleKwargs : dict
            Keyword arguments for resample supported by `mne.io.Epochs.resample`
            Please refer to mne for a detailed explanation.
        '''
        self.sfreq = sfreq
        self.mneResampleKwargs = mneResampleKwargs

    def __call__(self, input: dict) -> dict:
        for sub in input.values():
            if isinstance(sub, dict):
                for epoch in sub.values():
                    epoch.resample(self.sfreq, **self.mneResampleKwargs)
            else:
                sub.resample(self.sfreq, **self.mneResampleKwargs)
        return input

    def __repr__(self) -> str:
        s = f'Resample(sfreq={self.sfreq}'
        if self.mneResampleKwargs:
            s += f', {dict_to_str(self.mneResampleKwargs)}'
        return s + ')'
