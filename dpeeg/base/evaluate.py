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
from typing import Union
from torchmetrics.functional.classification.confusion_matrix import confusion_matrix


def save_cm_img(
    preds: torch.Tensor,
    target: torch.Tensor,
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
