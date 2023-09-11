#!/usr/bin/env python
# coding: utf-8

"""
    This module is used to evaluate the metrics of the model and save.

    @author: SheepTAO
    @data: 2023-4-13
"""


import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Union
from torchmetrics.functional.classification.confusion_matrix import confusion_matrix


def plt_cm_img(
    cm : Union[np.ndarray, torch.Tensor],
    classes : Union[list, tuple],
    imgPath : Optional[str] = None,
    store : bool = False
) -> Figure:
    '''Store the corresponding confusion matrix figure to the given img path
        or return a Axes instance.
            
    This function contains two functions, which are store a confusion matrix
    and return a Axes instance, determined by the parameter `store`.

    Parameters
    ----------
    cm : np.ndarray, torch.Tensor
        Used for plt confusion matrix figure.
    classes : list, tuple
        Labels corresponding to row and col.
    imgPath : str, optional
        Storing the figure to imgPath. Must be specified when store is true.
    store : bool, optional
        If store is True, this function will store a xx.png in the given path
        or will only return a Axes. Default is False.
    
    Returns
    -------
    Figure.
    '''
    dataFrame = pd.DataFrame(cm / cm.sum(1)[:, None], classes, classes) # type: ignore
    ax = sns.heatmap(dataFrame, annot=True, cmap='RdBu_r')
    
    # store image
    if store:
        plt.savefig(imgPath)
    
    return ax.get_figure()

def cal_cm_and_plt_img(
    preds : torch.Tensor,
    acts : torch.Tensor,
    classes : Union[list, tuple],
    imgPath : Optional[str] = None,
    store : bool = False
) -> Figure:
        '''Calculate confusion matrix and get (store) the corresponding image.

        Parameters
        ----------
        preds : torch.Tensor
            The predicted value of the model.
        acts : torch.Tensor
            The actual label of dataset.

        Other parameters refer to the function `store_cm_img`. Note classes is
        not optional.
        
        Returns
        -------
        Figure.
        '''
        cm = confusion_matrix(preds, acts, task='multiclass',
                              num_classes=len(classes))
        
        return plt_cm_img(cm, classes, imgPath, store)
