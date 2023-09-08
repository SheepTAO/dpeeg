#!/usr/bin/env python
# coding: utf-8

"""
    @Desc    : A Logger to implement terminal output and log file writing.
    @Author  : SheepTAO
    @Time    : 2023-07-20
"""

import os, sys, logging
from typing import Union, Optional


class Logger:
    def __init__(
        self,
        loger : Optional[str] = None,
        path : Optional[str] = None,
        mode : Optional[str] = 'w',
        clevel : Optional[Union[int, str]] = logging.DEBUG,
        flevel : Optional[Union[int, str]] = logging.INFO,
    ) -> None:
        '''Logging hooks for terminals and file streams.

        Can handle terminal printing and file writing in three situations:
        1.Terminal printing;
        2.Log file writing;
        3.Ordinary file writing.

        Parameters
        ----------
        loger : str, optional
            Specify the logger name, creating it if necessary. If no name is specified, 
            return the root logger. Default is None.
        path : str, optional
            The path of log file. Default is None.
        mode : str, optional
            The write mode of the log file. Default is 'w'.
        clevel : int, str, optional
            The log level of console. Default is DEBUG.
        flevel : int, str, optional
            The log level of filehandler. If it is None, the log file will not be written
            and parameter path will be ignored. Default is INFO.
        '''

        self._logger = logging.getLogger(loger)
        self._logger.setLevel(logging.DEBUG)

        sh = logging.StreamHandler()
        shfmt = logging.Formatter('%(message)s')
        sh.setFormatter(shfmt)
        sh.setLevel(clevel)
        self._logger.addHandler(sh)
        if flevel != None:
            assert path != None, 'path cannot be empty.'
            fh = logging.FileHandler(os.path.abspath(path), mode)
            fhfmt = logging.Formatter('[%(asctime)s] [%(levelname)8s]: %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
            fh.setFormatter(fhfmt)
            fh.setLevel(flevel)
            self._logger.addHandler(fh)

        sys.excepthook = self.handle_exception

    def debug(self, message : str):
        self._logger.debug(message)

    def info(self, message : str):
        self._logger.info(message)

    def warning(self, message : str):
        self._logger.warning(message)

    def error(self, message : str):
        self._logger.error(message)

    def critical(self, message : str):
        self._logger.critical(message)

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        self._logger.critical("Uncaught exception>>>", exc_info=(exc_type, exc_value, exc_traceback))
