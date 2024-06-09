#!/usr/bin/env python
# coding: utf-8

"""
    A stopping criteria class to compose any training termination condition.

    @Author  : SheepTAO
    @Time    : 2023-07-26
"""

import abc
import os, sys, json


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
    def __init__(self, stopcri : dict | str) -> None:
        '''The parent class for all the stop criteria.
        
        Parameters
        ----------
        stopcri : dict, str
            Type is str, indicate the configuration file path. Type is dict, 
            indicate a dictionary containing initialization parameters for the
            corresponding stopping criteria.
        
        Returns
        -------
        bool :
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
        If you wish to do and on multiple cases then do like: 
        And(And(A, B), C)...
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
    def __init__(self, max_epochs : int, var_name : str) -> None:
        '''Stop when given number of epochs reached.

        Parameters
        ----------
        max_epochs : int
            Maximum epochs to watch. 
        var_name : str
            Key name to compare with in the variables dictionary.
        '''
        self.max_epochs = max_epochs
        self.var_name = var_name

    def __call__(self, variables : dict) -> bool:
        return variables[self.var_name] >= self.max_epochs

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(max_epochs={self.max_epochs}'
                f', var_name={self.var_name})')


class NoDecrease(Criteria):
    def __init__(self, num_epochs : int, var_name : str, min_change = 1e-6):
        '''Stop if there is no decrease on a given monitor channel for given 
        number of epochs.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to wait while there is no decrease in the value.
        var_name : str
            Key name to compare with in the variables dictionary.
        min_change : float
            Minimum relative decrease which resets the num_epochs. 
        '''
        self.num_epochs = num_epochs
        self.var_name = var_name
        self.min_change = min_change
        self.min_value = float('inf')
        self.current_epoch = 0

    def __call__(self, variables : dict) -> bool:
        var = variables[self.var_name]
        devar = (1 - self.min_change) * self.min_value if self.min_value > 0 \
            else (1 + self.min_change) * self.min_value
        if var <= devar and var != 0:
            self.min_value= variables[self.var_name]
            self.current_epoch = 1
        else:
            self.current_epoch += 1

        return self.current_epoch >= self.num_epochs

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_epochs={self.num_epochs}'
                f', var_name={self.var_name}, min_change={self.min_change})')

    def reset_parameters(self) -> None:
        self.min_value = float('inf')
        self.current_epoch = 0


class Bigger(Criteria):
    def __init__(self, var : float, var_name : str) -> None:
        '''Stop when greater than the specified value.

        Parameters
        ----------
        var : float
            Maximum value to watch.
        var_name : str
            Key name to compare with in the variables dictionary.
        '''
        self.var = var
        self.var_name = var_name

    def __call__(self, variables: dict) -> bool:
        return variables[self.var_name] > self.var

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(var={self.var})'


class Smaller(Criteria):
    def __init__(self, var : float, var_name : str) -> None:
        '''Stop when less than the specified value.

        Parameters
        ----------
        var : float
            Minimum value to watch.
        var_name : str
            Key name to compare with in the variables dictionary.
        '''
        self.var = var
        self.var_name = var_name

    def __call__(self, variables: dict) -> bool:
        return variables[self.var_name] < self.var

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(var={self.var})'
