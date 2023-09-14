#!/usr/bin/env python
# coding: utf-8

"""
    Basic method of `transforms.py` file.
    
    @Author  : SheepTAO
    @Time    : 2023-09-14
"""


import os
import numpy as np
from numpy import ndarray
from typing import Optional, Union, Tuple
from ..tools.logger import loger, verbose


@verbose
def slide_win(
    data : ndarray,
    win : int = 125, 
    overlap : int = 0,
    label : Optional[ndarray] = None,
    verbose : Optional[Union[int, str]] = None,
) -> Union[Tuple[ndarray, ndarray], ndarray]:
    '''This transform is only splits the time series (dim = -1) through the sliding 
    window operation on the original dataset. If the time axis is not divisible by 
    the sliding window, the last remaining time data will be discarded.

    Parameters
    ----------
    data : array of float, shape (..., times)
        The data to split.
    win : int, optional
        The size of the sliding window. Default is 125.
    overlap : int, optional
        The amount of overlap between adjacent sliding windows. Default is 0.
    label : ndarray, optional
        The label of the data. If not None, label will update with sliding window.
        Default is None.

    Returns
    -------
    If label is not None, (data, label) will be returned, or data will be returned.
    '''
    if win < 0 or overlap < 0:
        raise ValueError('win and overlap only allow positive numbers, '
                         f'but got {win} and {overlap} respectively.')
    if overlap >= win:
        raise ValueError(f'overlap={overlap} should be less than win={win}.')

    loger.info(f'Sliding window with win:{win} and overlap:{overlap}.')
    winSize = win - overlap
    sldNum = data.shape[-1] // winSize
    if not sldNum:
        loger.warning('The sliding window is larger than the data to be split.')
    dataList = [data[..., : winSize]]
    for i in range(1, sldNum):
        sld = data[..., i*winSize-overlap :(i+1)*winSize-overlap]
        dataList.append(sld)

    data = np.concatenate(dataList)
    if isinstance(label, ndarray):
        label = np.repeat(label, sldNum)
        return data, label
    return data


@verbose
def save(
    folder : str,
    input : dict,
    verbose : Optional[Union[int, str]] = None,
) -> None:
    '''Save transformed dataset to a binary file in NumPy `.npy` format.

    Parameters
    ----------
    folder : str
        Folder name to save transformed data.
    input : dict
        Data to be saved.
    '''
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)
    if os.listdir(folder):
        raise FileExistsError(f'\'{folder}\' is not a empty folder.')
    
    loger.info(f'Transformed data will be saved in: \'{folder}\'')
    for sub, data in input.items():
        loger.info(f'Save transformed data of sub_{sub}.')
        for mode in ['train', 'test']:
            fileName = os.path.join(folder, f'{sub}_{mode}')
            np.save(fileName + '_data', data[mode][0])
            np.save(fileName + '_label', data[mode][1])


@verbose
def load(
    folder : str,
    subjects : Optional[list] = None,
    mode : str = 'all',
    verbose : Optional[Union[int, str]] = None,
) -> dict:
    '''Load saved transformed dataset from folder.

    Parameters
    ----------
    folder : str
        Folder name where transformed data is saved.
    subjects : list of int, optional
        List of subject number. If None, all subjects will be loaded. Default is None.
    mode : str, optional
        Mode to load data. If 'all', both train set and test set will be loaded. If 
        'train', only train set will be loaded. 'test' is the same.
    '''
    path = os.path.abspath(folder)
    loger.info(f'Loading dataset from \'{path}\'')

    pathList = os.listdir(path)
    subList = list(set([int(p.split('_')[0]) for p in pathList]))
    subList.sort()
    if subjects:
        intersection = set(subjects) & set(subList)
        exclude = set(subjects) - set(subList)
        if exclude:
            loger.warning(f'Could not find subjects: {exclude}, ' +
                          f'only load subjects: {intersection}')
        subList = list(intersection)

    dataset = {}
    for sub in subList:
        loger.info(f'Loading subject {sub}')
        fileName = os.path.join(path, str(sub))
        if mode.lower() == 'train' or mode.lower() == 'all':
            data = np.load(fileName + '_train_data.npy', allow_pickle=True)
            label = np.load(fileName + '_train_label.npy', allow_pickle=True)
            dataset.setdefault(sub, {})['train'] = [data, label]
        if mode.lower() == 'test' or mode.lower() == 'all':
            data = np.load(fileName + '_test_data.npy', allow_pickle=True)
            label = np.load(fileName + '_test_label.npy', allow_pickle=True)
            dataset.setdefault(sub, {})['test'] = [data, label]
        if mode not in ['train', 'test', 'all']:
            raise KeyError(f'Only support `all`, `train` and `test` mode.')
    loger.info('Load dataset done.')
    return dataset
