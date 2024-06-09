#!/usr/bin/env python
# coding: utf-8

"""
    General data processing tools.

    @Author  : SheepTAO
    @Time    : 2024-05-11
"""


from numpy import ndarray
from torch import Tensor
from typing import Literal


def check_data_label(
    data_label : list[Tensor | ndarray],
    sub : int | None = None,
    mode : str | None = None,
):
    '''Check data and label.

    Parameters
    ----------
    data_label : list of ndarray
        List containing data and labels.
    sub : int
        Number of subject.
    mode : str
        Mode of subject.
    '''
    error_msg = ''
    if sub:
        error_msg += f'subject {sub} '
    if mode:
        error_msg += f'{mode}set '
    
    assert len(data_label) == 2, f'{error_msg}miss data or label.'
    assert data_label[0].shape[0] == data_label[1].shape[0], \
        f'{error_msg}data and labels do not match in the first dimension.'


def check_sub_data(sub : int, sub_data) -> bool:
    '''Check the subject's data format.

    Check whether the subject's data format meets the requirements. If it does
    not meet the requirements, an error will be thrown. If there is no format
    error, a bool value is returned. True means that the dataset has been split,
    False means that the dataset has not been split yet.

    Parameters
    ----------
    sub : int
        Number of subject.
    sub_data : Any
        Subject's data.

    Returns
    -------
    bool
        True means splited, False means un-splited.
    '''
    # splited dataset
    if isinstance(sub_data, dict):
        split = True

        # trainset and testset
        if not sorted(sub_data.keys()) == ['test', 'train']:
            raise ValueError(f'Subject {sub} should have train and test set.')

        # data and label
        for mode, dl in sub_data.items(): # type: ignore
            if not (isinstance(dl, (list, tuple)) and len(dl) == 2):
                raise ValueError(f'Subject {sub} {mode}set miss data or label.')
    # un-splited dataset
    else:
        split = False

        # data and label
        if not(isinstance(sub_data, (list, tuple)) and len(sub_data) == 2):
            raise ValueError(f'Subject {sub} miss data or label.')

    return split


def check_dataset(dataset) -> bool:
    '''Check the format of the dataset.

    Determine whether the format of the data set meets the requirements of the 
    `dpeeg` data set format. If the requirements are not met, the corresponding
    error will be thrown. If the requirements are met, a bool value will be 
    returned. True means that the dataset has been split, False means that the 
    dataset has not been split yet.

    Parameters
    ----------
    dataset : Any
        Input dataset.

    Returns
    -------
    bool
        True means splited, False means un-splited.
    '''
    # entire dataset
    if not isinstance(dataset, dict):
        raise ValueError('Dataset should be dict.')

    if not all(isinstance(k, int) for k in dataset.keys()):
        raise ValueError('Name of subject should be int.')

    split = None
    for sub, sub_data in dataset.items():
        last_sub_split = split

        split = check_sub_data(sub, sub_data)

        if last_sub_split is not None and last_sub_split != split:
            raise ValueError(f'Subject {sub} format is not uniform.')

    return split # type: ignore


def total_trials(
    dataset : dict, 
    sub : int | None = None,
    mode : Literal['train', 'test', 'all'] = 'all'
) -> int:
    '''All trials in the dataset.

    Parameters
    ----------
    dataset : dict
        The dataset to be counted.
    sub : int, optional
        If None, count all subjects in the dataset.
    mode : str
        Mode to count data. If 'all', both trainset and testset will be counted.
        If 'train', only trainset will be counted. 'test' is the same.

    Returns
    -------
    int
        Total trials.
    '''
    train_trials = test_trials = 0

    for sub_id, sub_data in dataset.items():
        if sub is not None and sub_id == sub:
            train_trials += sub_data['train'][1].shape[0]
            test_trials += sub_data['test'][1].shape[0]

    if mode == 'train':
        return train_trials
    elif mode == 'test':
        return test_trials
    elif mode == 'all':
        return train_trials + test_trials
    else:
        raise KeyError(f'Only support `all`, `train` and `test` mode.')


def yield_data(
    input : dict,
    mode : Literal['train', 'test', 'all'] = 'all'
):
    '''Traverse all subject training and test set in the dataset, and return 
    the current subject name, mode and data.
    '''
    for sub, sub_data in input.items():
        for m in ['train', 'test']:
            if mode != 'all' and m != mode:
                continue
            yield sub, m, sub_data[m]