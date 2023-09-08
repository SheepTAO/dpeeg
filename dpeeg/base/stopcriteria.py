#!/usr/bin/env python
# coding: utf-8

"""
    A stop criteria class to compose any training termination condition.

    @Author  : SheepTAO
    @Time    : 2023-07-26
"""

import os, json
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, TypeAlias

StrPath: TypeAlias = str


class Criteria(ABC):
    @abstractmethod
    def __call__(self, variables : dict) -> bool:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def reset_variables(self) -> None:
        pass


class ComposeStopCriteria(Criteria):
    '''Advanced complex logical expression parser to compose stop criteria.
    '''
    def __init__(self, stopcri : Union[dict, StrPath]) -> None:
        '''
        stopcri : dict, str
            Type is str indicating the configuration file path.
            Type is dict, it should be similar to the following example:
         
            {
                'cri_1': {
                    'type': '',
                    'maxEpochs' : None,
                    'varName' : None,
                },
                'sym_2' : 'AND',
                'cri_3' : {
                    'cri_1' : {
                        'type' : '',
                        'maxEpochs' : 1,
                    },
                    'sym_2' : 'or',
                    'cri_3' : {
                        'type' : '',
                        'parameter' : '',
                    }
                },
                'cri_4' : {
                    ...
                },
                ...
            }

            which 'cri_x' means x_th stop criteria, 'sym_x' indicates x_th logic 
            symbol and nested logical experssion is allowed. Therefore, the logical
            expression of the above configuration is
            cri_1 and (cri[_3]_1 or cri[_3]_3)
        '''
        if isinstance(stopcri, StrPath):
            with open(os.path.abspath(stopcri)) as f:
                stopcri = json.load(f)
                
        self.stopcri = stopcri
        self.stack = []
        self.repr = ''
        self.call = self._parser_cri(stopcri)

    def _has_nested_dict(self, dictionary : dict):
        for value in dictionary.values():
            if isinstance(value, dict):
                return True
        return False

    def _get_cri(self, value : dict):
        if value['type'].upper() in CURRENT_MODULE:
            kwargs = value.copy()
            kwargs.pop('type')
            return CURRENT_MODULE[value['type'].upper()](**kwargs)
        raise KeyError(f'Criteria `{type}` is not supported.')

    def _logical_expression_call(
        self, 
        cri1 : Callable, 
        sym : Optional[str] = None, 
        cri2 : Optional[Callable] = None
    ) -> Callable:
        if sym == None:
            return cri1
        if sym.upper() in CURRENT_MODULE:
            return CURRENT_MODULE[sym.upper()](cri1, cri2)
        raise KeyError(f'Logic symbols `{sym}` is not supported.')

    def _parser_cri(self, stopcri : dict) -> Callable[[dict], bool]:
        scLen = len(stopcri)
        assert scLen % 2 != 0, 'Count of criteria should be an even.'

        for idx in range(1, scLen + 1):
            if idx % 2:
                # parser criteria
                value = stopcri[f'cri_{idx}']
                if self._has_nested_dict(value):
                    self.repr += '('
                    self._parser_cri(value)
                    self.repr += ')'
                else:
                    cri = self._get_cri(value)
                    self.repr += str(cri)
                    self.stack.append(cri)
            else:
                # parser logic symbols
                sym = stopcri[f'sym_{idx}']
                self.repr += f' {sym.upper()} '
                self.stack.append(sym)

            # update logical expression
            if len(self.stack) == 3:
                cri1 = self.stack.pop()
                sym  = self.stack.pop()
                cri2 = self.stack.pop()
                self.stack.append(self._logical_expression_call(cri1, sym, cri2))
        
        return self.stack[-1]

    def __call__(self, variables: dict) -> bool:
        return self.call(variables)

    def __repr__(self) -> str:
        return self.repr

    def reset_variables(self) -> None:
        self.call.reset_variables()

class And(Criteria):
    def __init__(self, cri1 : Criteria, cri2 : Criteria) -> None:
        self.cri1 = cri1
        self.cri2 = cri2

    def __call__(self, variables: dict) -> bool:
        return self.cri1(variables) and self.cri2(variables)

    def __repr__(self) -> str:
        return 'AND'

    def reset_variables(self) -> None:
        self.cri1.reset_variables()
        self.cri2.reset_variables()


class Or(Criteria):
    def __init__(self, cri1 : Criteria, cri2 : Criteria) -> None:
        self.cri1 = cri1
        self.cri2 = cri2

    def __call__(self, variables: dict) -> bool:
        return self.cri1(variables) or self.cri2(variables)

    def __repr__(self) -> str:
        return 'OR'

    def reset_variables(self) -> None:
        self.cri1.reset_variables()
        self.cri2.reset_variables()


class MaxEpoch(Criteria):
    '''Stop when given number of epochs reached.
    '''
    def __init__(self, maxEpochs : int, varName : str) -> None:
        '''
        maxEpochs : int
            Maximum epochs to watch. 
        varName : str
            Key name to compare with in the variables dictionary
        '''
        self.maxEpochs = maxEpochs
        self.varName = varName

    def __call__(self, variables : dict) -> bool:
        return variables[self.varName] >= self.maxEpochs

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(maxEpochs={self.maxEpochs}'
                f', varName={self.varName})')

    def reset_variables(self) -> None:
        return super().reset_variables()


class NoDecrease(Criteria):
    '''Stop if there is no decrease on a given monitor channel for given number of epochs.
    '''
    def __init__(self, numEpochs : int, varName : str, minChange = 1e-6) -> None:
        '''
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
        if variables[self.varName] < (1 - self.minChange) * self.minValue:
            self.minValue= variables[self.varName]
            self.currentEpoch = 1
        else:
            self.currentEpoch += 1

        return self.currentEpoch >= self.numEpochs

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(numEpochs={self.numEpochs}'
                f', varName={self.varName}, minChange={self.minChange})')

    def reset_variables(self) -> None:
        self.minValue = float('inf')
        self.currentEpoch = 0

CURRENT_MODULE = {
    'AND': And,
    'OR': Or,
    'MAXEPOCH': MaxEpoch,
    'NODECREASE': NoDecrease,
}
