#!/usr/bin/env python
# coding: utf-8

"""
    This module is used to evaluate the metrics of the model and save.

    @author: SheepTAO
    @data: 2023-4-13
"""


import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Union, List, Optional
from torchmetrics.functional.classification.confusion_matrix import confusion_matrix


def save_cm_img(
    preds : Tensor,
    target : Tensor,
    clsName : Union[list, tuple],
    figPath : str,
) -> None:
    '''Calculate and save the corresponding confusion matrix figure to the given
    img path.

    Parameters
    ----------
    preds : Tensor
        Predicted labels, as returned by a classifier.
    target : Tensor
        Ground truth (correct) labels.
    clsName : list, tuple
        The name of dataset labels.
    figPath : str
        Path to store the figure.
    '''
    cm = confusion_matrix(preds, target, 'multiclass', num_classes=len(clsName))
    fig = sns.heatmap(cm / cm.sum(1, keepdim=True), annot=True, cmap='Blues',
                      xticklabels=clsName, yticklabels=clsName) # type: ignore
    plt.savefig(figPath)
    # !!! Clear the current figure
    plt.clf()

def ttest_corrected(
    score1 : Tensor,
    score2 : Tensor,
    n1 : int,
    n2 : int,
) -> None:
    '''Corrected Paired t test for comparing the performance of two models.

    Parameters
    ----------
    score1 : 1-D Tensor
        Results of model A of shape (N, ).
    score2 : 1-D Tensor
        Results of model B of shape (N, ).
    n1 : int
        The number of data points used for training.
    n2 : int
        The number of data points used for testing.
    
    Returns
    -------
    t : float
        The t-statistic.
    pvalue : float
        Two-tailed p-value. If the chosen significance level is larger than the
        p-value, we reject the null hypothesis and accept that there are signi-
        ficant differences in the two compared models.

    Notes
    -----
    heavy development
    '''
    if score1.dim() != 1:
        raise ValueError('The input tensor dimension should be 1, but got '
                         f'{score1.dim()} of score1.')
    if score2.dim() != 1:
        raise ValueError('The input tensor dimension should be 1, but got '
                         f'{score2.dim()} of score2.')

    diff = score1 - score2
    dbar = torch.mean(diff)
    sigma2 = torch.var(diff)
    sigma2Mod = sigma2 * (1/(n1+n2) + (n2/n1))

    tStatic = dbar / torch.sqrt(sigma2Mod)
