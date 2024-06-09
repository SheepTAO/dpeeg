#!/usr/bin/env python
# coding: utf-8

"""
    @Desc    : A Logger to implement terminal output and log file writing.
    @Author  : SheepTAO
    @Time    : 2023-07-20
"""


import os, sys, logging


_Level = int | str | None


class Logger:
    def __init__(
        self,
        loger : str = 'dpeeg_root',
        path : str | None = None,
        mode : str = 'a',
        clevel : int | str = logging.INFO,
        flevel : _Level = None,
    ) -> None:
        '''Logging hooks for terminals and file streams.

        Can handle terminal printing and file writing in three situations:
        1.Terminal print;
        2.Terminal and file streams write simultaneously;
        3.Ordinary file write.

        Parameters
        ----------
        loger : str
            Specify the logger name, creating it if necessary. If no name is specified, 
            return the root logger.
        path : str, optional
            The path of log file.
        mode : str
            The write mode of the log file.
        clevel : int, str
            The log level of console.
        flevel : int, str, optional
            The log level of filehandler. If it is None, the log file will not be written
            and parameter path will be ignored.
        '''

        self._logger = logging.getLogger(loger)
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)

        self._sh = logging.StreamHandler()
        shfmt = logging.Formatter('%(message)s')
        self._sh.setFormatter(shfmt)
        self._sh.setLevel(clevel)
        self._logger.addHandler(self._sh)
        if flevel != None:
            assert path != None, 'path cannot be empty.'
            self._fh = logging.FileHandler(os.path.abspath(path), mode)
            fhfmt = logging.Formatter('[%(asctime)s] [%(levelname)8s]: %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
            self._fh.setFormatter(fhfmt)
            self._fh.setLevel(flevel)
            self._logger.addHandler(self._fh)

        sys.excepthook = self.handle_exception

    def update_sh_level(self, level : int | str):
        self._sh.setLevel(level)

    def get_sh_level(self) -> str:
        return logging.getLevelName(self._sh.level)

    def debug(self, message : str):
        self._logger.debug(f'[DEBUG]: {message}')

    def info(self, message : str):
        self._logger.info(message)

    def warning(self, message : str):
        self._logger.warning(f'[WARNING]: {message}')

    def error(self, message : str):
        self._logger.error(f'[ERROR]: {message}')

    def critical(self, message : str):
        self._logger.critical(f'[CRITICAL]: {message}')

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        self._logger.critical("Uncaught exception>>>", exc_info=(exc_type, exc_value, exc_traceback))
