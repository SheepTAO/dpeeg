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

from ..utils import dict_to_str, unpacked


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
        *preprocess : Preprocess, 
    ) -> None:
        '''Compose serval preprocess together.

        The preprocess in `ComposePreprocess` are connected in a cascading way.

        Parameters
        ----------
        preprocess : sequential container of Preprocess
            Preprocess (sequential of `Preprocess` objects): sequential of pre-
            process to compose.
        '''
        self.preps : List[Preprocess] = []
        self.appends(*preprocess)

    def __call__(self, input: dict) -> dict:
        for t in self.preps:
            input = t(input)
        return input

    def __repr__(self) -> str:
        s = 'ComposePreprocess('
        if len(self.preps) == 0:
            return s + ')'
        else:
            for idx, pre in enumerate(self.preps):
                s += f'\n ({idx}): {pre}'
        return s + '\n)'

    def appends(self, *preprocess : Preprocess) -> None:
        '''Append preprocess to the last of composes.
        '''
        preps = unpacked(*preprocess)
        for prep in preps:
            if isinstance(prep, ComposePreprocess):
                self.preps.extend(prep.get_data())
            else:
                self.preps.append(prep)

    def insert(self, index : int, preprocess : Preprocess) -> None:
        '''Insert a preprocess at index.
        '''
        self.preps.insert(index, preprocess)

    def get_data(self) -> List[Preprocess]:
        '''Return list of Preprocess.
        '''
        return self.preps


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
