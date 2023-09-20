#!/usr/bin/env python
# coding: utf-8

"""
    Basic method of `transforms.py` file.
    
    @Author  : SheepTAO
    @Time    : 2023-09-14
"""


import os
import torch
import numpy as np
from numpy import ndarray
from torch import Tensor
from typing import Optional, Union, Tuple, List

from ..utils import loger, verbose, DPEEG_SEED


@verbose
def split_train_test(
    *arrs, 
    testSize : float = .2, 
    seed : int = DPEEG_SEED, 
    sample : Optional[List[int]] = None, 
    verbose : Optional[Union[int, str]] = None
) -> list:
    '''Split an dataset into training and testing sets. The axis along which
    to split is 0.

    Parameters
    ----------
    *arrs : sequence of indexables with same length / shape[0]
        Allowed inputs are lists and numpy arrays.
    testSize : float
        The proportion of the test set. Default is 0.2. If index is not None,
        testSize will be ignored. Default use stratified fashion and the last
        arr serves as the class labels.
    seed : int
        Random seed when splitting. Default is DPEEG_SEED.
    sample : list of int, optional
        A list of integers, the entries indicate which data were selected
        as the test set. If None, testSize will be used. Default is None.
    '''
    nArrs = len(arrs)
    if nArrs == 0:
        raise ValueError('At least one array required as input.')

    arrList = [np.array(arr) for arr in arrs]
    lengths = [len(arr) for arr in arrList]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            'Found input variables with inconsistent numbers of samples: %r'
            % [int(l) for l in lengths]
        )
        
    if sample:
        smparr = np.array(sample)
        if np.unique(smparr).size != smparr.size:
            raise IndexError(
                f'Found repeated sampling of test set: {smparr.tolist()}.'
            )
        length = lengths[0]
        if smparr.size >= uniques:
            raise IndexError(f'The number of samples (={len(smparr)}) in the '
                             'test set cannot exceed the total number of data '
                             f'sets (={length}).'
            )

        testSample, trainSample = smparr, np.setdiff1d(np.arange(length), smparr)
        res = []
        for arr in arrList:
            res.extend([arr[trainSample], arr[testSample]])
        return res
    else:
        from sklearn.model_selection import train_test_split
        return train_test_split(arrList, test_size=testSize, random_state=seed,
                                stratify=arrList[-1])


@verbose
def to_tensor(
    data: Union[Tensor, ndarray],
    label: Union[Tensor, ndarray],
    verbose : Optional[Union[int, str]] = None
) -> Tuple[Tensor, Tensor]:
    '''Convert the numpy data and label into trainable Tensor format.
    '''
    dataT = torch.from_numpy(data).float() \
        if isinstance(data, ndarray) else data.float()
    labelT = torch.from_numpy(label).long() \
        if isinstance(label, ndarray) else label.long()
    if dataT.size(0) != labelT.size(0):
        loger.warning('Data and label do not match in the first dimension: '
                      f'{dataT.size(0)} and {labelT.size(0)} respectively.')
    return dataT, labelT


@verbose
def slide_win(
    data : ndarray,
    win : int, 
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
    win : int
        The size of the sliding window.
    overlap : int
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
    subjects : Optional[List[int]] = None,
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
    mode : str
        Mode to load data. If 'all', both train set and test set will be loaded. If 
        'train', only train set will be loaded. 'test' is the same. Default is 'all'.
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
