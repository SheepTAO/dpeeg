#!/usr/bin/env python
# coding: utf-8

"""
    Commonly used models for decoding EEG signals.
    
    @Author  : SheepTAO
    @Time    : 2023-11-05
"""


import torch
import torch.nn as nn


class Conv2dWithNorm(nn.Conv2d):
    def __init__(self, *args, do_weight_norm=True, max_norm=1., p=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(
                self.weight.data, self.p, 0, self.max_norm
            )
        return super().forward(input)

    def __repr__(self):
        repr = super().__repr__()
        if self.do_weight_norm:
            last_bracket_index = repr.rfind(')')
            self_repr = f', max_norm={self.max_norm}, p={self.p}'
            repr = repr[:last_bracket_index] + self_repr + ')'
        return repr


class LinearWithNorm(nn.Linear):
    def __init__(self, *args, do_weight_norm=True, max_norm=1., p=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(
                self.weight.data, self.p, 0, self.max_norm
            )
        return super().forward(input)

    def __repr__(self):
        repr = super().__repr__()
        if self.do_weight_norm:
            last_bracket_index = repr.rfind(')')
            self_repr = f', max_norm={self.max_norm}, p={self.p}'
            repr = repr[:last_bracket_index] + self_repr + ')'
        return repr
