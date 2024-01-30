#!/usr/bin/env python
# coding: utf-8

"""
    @Desc    : A file handler to write in realtime.
    @Author  : SheepTAO
    @Time    : 2023-07-25
"""

import os
import csv


class Filer:
    def __init__(
        self, 
        path : str, 
        mode : str = 'a', 
        exist_ok : bool = False
    ) -> None:
        '''A simple file handler to write in realtime.

        Parameters
        ----------
        path : str
            File path. Support hierarchical folder structure.
        mode : str
            Mode of write.
        exist_ok : bool
            Whether to allow files that already exist.
        '''
        self.path = path
        self.mode = mode
        if os.path.exists(path) and not exist_ok:
            raise FileExistsError(f"File '{path}' already exists.")
        folder_path = os.path.dirname(path)
        os.makedirs(folder_path, exist_ok=True)

    def write(self, data : str) -> None:
        with open(self.path, self.mode) as f:
            f.write(data)


class CSVer(Filer):
    def __init__(self, path: str, mode: str = 'a') -> None:
        super().__init__(path, mode)

    def write(self, data: str) -> None:
        with open(self.path, self.mode) as f:
            w = csv.writer(f)
            w.writerows(data)
