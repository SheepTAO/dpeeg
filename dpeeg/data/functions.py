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
from ..tools.logger import _Level


@verbose
def split_train_test(
    *arrs, 
    test_size : float = .25, 
    seed : int = DPEEG_SEED, 
    sample : Optional[List[int]] = None, 
    verbose : _Level = None
) -> list:
    '''Split an dataset into training and testing sets. The axis along which
    to split is 0.

    Parameters
    ----------
    *arrs : sequence of indexables with same length / shape[0]
        Allowed inputs are lists and numpy arrays.
    test_size : float
        The proportion of the test set. Default is 0.25. If index is not None,
        test_size will be ignored. Default use stratified fashion and the last
        arr serves as the class labels.
    seed : int
        Random seed when splitting. Default is DPEEG_SEED.
    sample : list of int, optional
        A list of integers, the entries indicate which data were selected
        as the test set. If None, test_size will be used. Default is None.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    '''
    num_arrs = len(arrs)
    if num_arrs == 0:
        raise ValueError('At least one array required as input.')

    arr_list = [np.array(arr) for arr in arrs]
    lengths = [len(arr) for arr in arr_list]
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
        for arr in arr_list:
            res.extend([arr[trainSample], arr[testSample]])
        return res
    else:
        from sklearn.model_selection import train_test_split
        return train_test_split(
            *arr_list, test_size=test_size, random_state=seed, 
            stratify=arr_list[-1]
        )


@verbose
def to_tensor(
    data: Union[Tensor, ndarray],
    label: Union[Tensor, ndarray],
    verbose : _Level = None
) -> Tuple[Tensor, Tensor]:
    '''Convert the numpy data and label into trainable Tensor format.
    '''
    # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    dataT = torch.from_numpy(np.ascontiguousarray(data)).float() \
        if isinstance(data, ndarray) else data.float()
    labelT = torch.from_numpy(np.ascontiguousarray(label)).long() \
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
    verbose : _Level = None,
) -> Tuple[ndarray, Optional[ndarray]]:
    '''This transform is only splits the time series (dim = -1) through the 
    sliding window operation on the original dataset. If the time axis is not
    divisible by the sliding window, the last remaining time data will be 
    discarded.

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
    data : ndarray
        Data after sliding window.
    label : ndarray, None
        If label is None, return None. The label corresponding to each window.
    '''
    if win < 0 or overlap < 0:
        raise ValueError('win and overlap only allow positive numbers, '
                         f'but got {win} and {overlap} respectively.')
    if overlap >= win:
        raise ValueError(f'overlap={overlap} should be less than win={win}.')

    loger.info(f'Sliding window with win:{win} and overlap:{overlap}.')

    end = win
    times = data.shape[-1]
    if end > times:
        loger.warning('The window is larger than the times to be split.')
        if isinstance(label, ndarray):
            return data, label
        return data, None

    sld_num = 0
    data_list = []
    while end <= times:
        data_list.append(data[..., end-win:end])
        loger.info(f' Intercept data from {end-win} to {end}.')
        sld_num += 1
        end += win - overlap

    data = np.concatenate(data_list)
    if isinstance(label, ndarray):
        label = np.repeat(label, sld_num)
        return data, label
    return data, None


@verbose
def segmentation_and_reconstruction(
    data : ndarray,
    label : ndarray,
    multiply : int = 1,
    n : int = 8,
) -> Tuple[ndarray, ndarray]:
    '''Signal Segmentation and Recombination in Time Domain.

    This approach is to first divide each training EEG trial into several 
    segments and then generate new artificial trials as a concatenation of 
    segments coming from different and randomly selected training trials from
    the same class while maintaining the original time order.

    Parameters
    ----------
    data : ndarray
        Data that will be segmented and randomly recombined. Shape as `(N, ..., 
        T)`, with `N` the number of data and `T` the number of samples.
    label : ndarray
        The label corresponding to the data. Shape as `(N)`.
    multiply : int
        The data will be enhanced several times. Default is 1.
    n : int
        The data will be segmented into n parts in time domain. And n should be
        evenly divisible by the time length of the data. Default is 8. For 
        example, 250Hz data has 1,000 sampling points, and cutting it into 8 
        parts means that each segment is 0.5s of data.

    Notes
    -----
    The function does not judge the multiples or the number of segmentation.

    References
    ----------
    F. Lotte, “Signal Processing Approaches to Minimize or Suppress 
    Calibration Time in Oscillatory Activity-Based Brain–Computer Interfaces,
    ” Proc. IEEE, vol. 103, no. 6, pp. 871–890, Jun. 2015, 
    doi: 10.1109/JPROC.2015.2404941.
    '''
    if n == 0:
        raise ValueError('Parameter n must be at least 1.')
    
    if data.shape[-1] % n:
        raise ValueError(f'The data time length ({data.shape[-1]}) should be '
                         f'a multiple of n={n}.')

    win = data.shape[-1] // n
    aug_data, aug_label = [], []

    for lb in np.unique(label):
        idx = np.where(label == lb)
        tmp_data, tmp_label = data[idx], label[idx]
        m = tmp_data.shape[0] * multiply

        tmp_aug_data = np.empty((m, *data.shape[1:]))
        for i in range(m):
            for j in range(n):
                randIdx = np.random.randint(0, tmp_data.shape[0], n)
                tmp_aug_data[i, ..., j * win : (j + 1) * win] = \
                    tmp_data[randIdx[j] , ..., j * win : (j + 1) * win]

        aug_data.append(tmp_aug_data)
        aug_label.append(np.repeat(lb, m))
        aug_data.append(tmp_data)
        aug_label.append(tmp_label)

    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    shuffle = np.random.permutation(aug_data.shape[0])
    aug_data = aug_data[shuffle]
    aug_label = aug_label[shuffle]

    return aug_data, aug_label


@verbose
def save(
    folder : str,
    input : dict,
    verbose : _Level = None,
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
            file_name = os.path.join(folder, f'{sub}_{mode}')
            np.save(file_name + '_data', data[mode][0])
            np.save(file_name + '_label', data[mode][1])


@verbose
def load(
    folder : str,
    subjects : Optional[List[int]] = None,
    mode : str = 'all',
    verbose : _Level = None,
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

    path_list = os.listdir(path)
    sub_list = list(set([int(p.split('_')[0]) for p in path_list]))
    sub_list.sort()
    if subjects:
        intersection = set(subjects) & set(sub_list)
        exclude = set(subjects) - set(sub_list)
        if exclude:
            loger.warning(f'Could not find subjects: {exclude}, ' +
                          f'only load subjects: {intersection}')
        sub_list = list(intersection)

    dataset = {}
    for sub in sub_list:
        loger.info(f'Loading subject {sub}')
        file_name = os.path.join(path, str(sub))

        if mode.lower() == 'train' or mode.lower() == 'all':
            data = np.load(file_name + '_train_data.npy', allow_pickle=True)
            label = np.load(file_name + '_train_label.npy', allow_pickle=True)
            dataset.setdefault(sub, {})['train'] = [data, label]

        if mode.lower() == 'test' or mode.lower() == 'all':
            data = np.load(file_name + '_test_data.npy', allow_pickle=True)
            label = np.load(file_name + '_test_label.npy', allow_pickle=True)
            dataset.setdefault(sub, {})['test'] = [data, label]

        if mode not in ['train', 'test', 'all']:
            raise KeyError(f'Only support `all`, `train` and `test` mode.')
    loger.info('Load dataset done.')
    return dataset
