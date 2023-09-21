#!/usr/bin/env python
# coding: utf-8

"""
    A stopping criteria class to compose any training termination condition.

    @Author  : SheepTAO
    @Time    : 2023-07-26
"""

import abc
import os, sys, json
from typing import Union


CURRENT_MODULE = sys.modules[__name__]


class Criteria(abc.ABC):
    @abc.abstractmethod
    def __call__(self, variables : dict) -> bool:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    def reset_parameters(self) -> None:
        pass


class ComposeStopCriteria(Criteria):
    '''Advanced complex logical expression parser to compose stopping criteria.
    '''
    def __init__(self, stopcri : Union[dict, str]) -> None:
        '''The parent class for all the stop criteria.
        
        Parameters
        ----------
        stopcri : dict, str
            Type is str, indicate the configuration file path. Type is dict, 
            indicate a dictionary containing initialization parameters for the
            corresponding stopping criteria.
        
        Returns
        -------
            Whether the stopping condition is reached.
        '''
        if isinstance(stopcri, str):
            with open(os.path.abspath(stopcri)) as fp:
                cri : dict = json.load(fp)
        else:
            cri : dict = stopcri
        self.call : Criteria = CURRENT_MODULE.__dict__[list(cri.keys())[0]] \
            (**cri[list(cri.keys())[0]])

    def __call__(self, variables: dict) -> bool:
        return self.call(variables)

    def __repr__(self) -> str:
        return str(self.call)

    def reset_parameters(self) -> None:
        self.call.reset_parameters()


class And(Criteria):
    def __init__(self, cri1, cri2) -> None:
        '''And operation on two stop criteria.

        Parameters
        ----------
        cri1 : dict
            Dictionary describing first criteria.
        cri2 : dict
            Dictionary describing second criteria.

        Notes
        -----
        If you wish to do and on multiple cases then do like: And(And(A, B), C)...
        '''
        self.cri1 : Criteria = CURRENT_MODULE.__dict__[list(cri1.keys())[0]] \
            (**cri1[list(cri1.keys())[0]])
        self.cri2 : Criteria = CURRENT_MODULE.__dict__[list(cri2.keys())[0]] \
            (**cri2[list(cri2.keys())[0]])

    def __call__(self, variables: dict) -> bool:
        return self.cri1(variables) and self.cri2(variables)

    def __repr__(self) -> str:
        return f'({self.cri1} \'AND\' {self.cri2})'

    def reset_parameters(self) -> None:
        self.cri1.reset_parameters()
        self.cri2.reset_parameters()


class Or(Criteria):
    def __init__(self, cri1 : dict, cri2 : dict) -> None:
        '''Or operation on two stop criteria.

        Parameters
        ----------
        cri1 : dict
            Dictionary describing first criteria.
        cri2 : dict
            Dictionary describing second criteria.

        Notes
        -----
        If you wish to do and on multiple cases then do like: And(And(A, B), C)...
        '''
        self.cri1 : Criteria = CURRENT_MODULE.__dict__[list(cri1.keys())[0]] \
            (**cri1[list(cri1.keys())[0]])
        self.cri2 : Criteria = CURRENT_MODULE.__dict__[list(cri2.keys())[0]] \
            (**cri2[list(cri2.keys())[0]])

    def __call__(self, variables: dict) -> bool:
        return self.cri1(variables) or self.cri2(variables)

    def __repr__(self) -> str:
        return f'({self.cri1} \'OR\' {self.cri2})'

    def reset_parameters(self) -> None:
        self.cri1.reset_parameters()
        self.cri2.reset_parameters()


class MaxEpoch(Criteria):
    def __init__(self, maxEpochs : int, varName : str) -> None:
        '''Stop when given number of epochs reached.

        Parameters
        ----------
        maxEpochs : int
            Maximum epochs to watch. 
        varName : str
            Key name to compare with in the variables dictionary.
        '''
        self.maxEpochs = maxEpochs
        self.varName = varName

    def __call__(self, variables : dict) -> bool:
        return variables[self.varName] >= self.maxEpochs

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(maxEpochs={self.maxEpochs}'
                f', varName={self.varName})')


class NoDecrease(Criteria):
    def __init__(self, numEpochs : int, varName : str, minChange = 1e-6) -> None:
        '''Stop if there is no decrease on a given monitor channel for given 
        number of epochs.

        Parameters
        ----------
        numEpochs : int
            Number of epochs to wait while there is no decrease in the value.
        varName : str
            Key name to compare with in the variables dictionary.
        minChange : float
            Minimum relative decrease which resets the numEpochs. Default is 1e-6.
        '''
        self.numEpochs = numEpochs
        self.varName = varName
        self.minChange = minChange
        self.minValue = float('inf')
        self.currentEpoch = 0

    def __call__(self, variables : dict) -> bool:
        if variables[self.varName] <= (1 - self.minChange) * self.minValue:
            self.minValue= variables[self.varName]
            self.currentEpoch = 1
        else:
            self.currentEpoch += 1

        return self.currentEpoch >= self.numEpochs

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(numEpochs={self.numEpochs}'
                f', varName={self.varName}, minChange={self.minChange})')

    def reset_parameters(self) -> None:
        self.minValue = float('inf')
        self.currentEpoch = 0


class Bigger(Criteria):
    def __init__(self, var : float, varName : str) -> None:
        '''Stop when greater than the specified value.

        Parameters
        ----------
        var : float
            Maximum value to watch.
        varName : str
            Key name to compare with in the variables dictionary.
        '''
        self.var = var
        self.varName = varName

    def __call__(self, variables: dict) -> bool:
        return variables[self.varName] > self.var

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(var={self.var})'


class Smaller(Criteria):
    def __init__(self, var : float, varName : str) -> None:
        '''Stop when less than the specified value.

        Parameters
        ----------
        var : float
            Minimum value to watch.
        varName : str
            Key name to compare with in the variables dictionary.
        '''
        self.var = var
        self.varName = varName

    def __call__(self, variables: dict) -> bool:
        return variables[self.varName] < self.var

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(var={self.var})'
