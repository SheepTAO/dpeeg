#!/usr/bin/env python
# coding: utf-8

"""
    Provide some commonly used evaluation index calculations.

    Heavy development!

    @Author  : SheepTAO
    @Time    : 2024-04-29
"""


import torch
from torch import Tensor


class AggMetrics:
    def __init__(self, device : torch.device | str = 'cpu') -> None:
        '''Aggregate tensors.
        '''
        self.device = torch.device(device)
        self.value = torch.empty(0, device=self.device)
    
    def update(self, value : Tensor):
        '''Update new data.

        Parameters
        ----------
        value : Tensor
            Additional tensor dimensions will be flattened.
        '''
        self.value = torch.cat([self.value, torch.flatten(value).to(self.device)])

    def mean(self) -> Tensor:
        return self.value.mean()

    def std(self, correction : int = 0) -> Tensor:
        return self.value.std(correction=correction)

    def cat(self) -> Tensor:
        return self.value