#!/usr/bin/env python
# coding: utf-8

"""
    @Desc    : A simple timer to record the running time of the program.
    @Author  : SheepTAO
    @Time    : 2023-07-24
"""


import time
from typing import Tuple, Optional


class Timer:
    def __init__(self) -> None:
        '''Timing the running time of the program.
        Initialization will automatically start a default timer.
        '''
        self._beg = {}
        self._end = {}
        self.start(__name__)

    def start(self, name : Optional[str] = __name__) -> str:
        '''Update specified timer. Create a timer if it does not exist.
        '''
        self._beg[name] = time.time()
        return self.ctime()

    def stop(self, name : Optional[str] = __name__) -> Tuple[int, int, float]:
        '''Get the time of the specified timer.
        '''
        if name not in self._beg.keys():
            raise RuntimeError(f'Timer `{name}` has not stared yet.')
        self._end[name] = time.time()
        sec = self._end[name] - self._beg[name]
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)

        return int(h), int(m), s

    @staticmethod
    def ctime() -> str:
        return time.ctime()
