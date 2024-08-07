#!/usr/bin/env python
# coding: utf-8

"""
    @Desc    : A simple timer to record the running time of the program.
    @Author  : SheepTAO
    @Time    : 2023-07-24
"""


import time


class Timer:
    '''Timing the running time of the program.

    Examples
    --------
    If you want to start a default timer and then calculate the program interval,
    you can do like:
    >>> timer = Timer()
    >>> do_something()
    >>> h, m, s = timer.stop()
    >>> h, m, s
    1, 2, 13.435
        
    And you can update the start time of the default timer, just do like:
    >>> timer.start()

    You can also register a specified timer and update the start time of it.
    >>> timer.start('new')
    >>> h, m, s = timer.stop('new')
    '''
    def __init__(self) -> None:
        '''Timing the running time of the program.
        Initialization will automatically start a default timer (dpeeg_timer).
        '''
        self._beg = {}
        self.start('dpeeg_timer')

    def start(self, name : str = 'dpeeg_timer') -> str:
        '''Update specified timer or register a timer if it does not exist.
        '''
        self._beg[name] = time.time()
        return self.ctime()

    def stop(self, name : str = 'dpeeg_timer') -> tuple[int, int, float]:
        '''Get the hours, minutes, and seconds of a specific timer.
        '''
        sec = self.stop_precise(name)
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)

        return int(h), int(m), s

    def stop_precise(self, name : str = 'dpeeg_timer') -> float:
        '''Get the precise time for a specific timer.
        '''
        if name not in self._beg.keys():
            raise RuntimeError(f'Timer `{name}` has not been registered yet.')
        ct = time.time()
        return ct - self._beg[name]

    @staticmethod
    def ctime() -> str:
        return time.ctime()
