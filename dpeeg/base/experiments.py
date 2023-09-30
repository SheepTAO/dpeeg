#!/usr/bin/env python
# coding: utf-8

"""
    This module defines with what experiments the module is trained and validated.

    experiments:
        1 - kFoldCV : K-Fold cross-validation.
        2 - HoldOut : hold out validation.

    @Author  : SheepTAO
    @Time    : 2023-07-26
"""


import os, pickle
from torch import Tensor
from typing import Union, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from torchmetrics.functional.classification.cohen_kappa import cohen_kappa
from torchmetrics.aggregation import MeanMetric, CatMetric

from .train import Train
from ..data.datasets import EEGDataset
from ..tools import Timer, Filer, Logger
from ..utils import DPEEG_SEED, DPEEG_DIR
from .evaluate import save_cm_img


class KFoldCV:
    def __init__(
        self,
        trainer : Train,
        k : int = 10,
        outFolder : Optional[str] = None,
        maxEpochs_1 : int = 1500,
        maxEpochs_2 : int = 600,
        noIncreaseEpochs : int = 100,
        varCheck : str = 'valInacc',
        shuffle : bool = True,
        seed : int = DPEEG_SEED,
        verbose : Union[int, str] = 'INFO',
    ) -> None:
        '''K-Fold cross-validation class.

        Parameters
        ----------
        trainer : Train
            Trainer used for training module on dataset.
        k : int, optional
            k of k-Fold. Default is 10.
        outFolder : str, optional
            Store all experimental results in a folder named with the model 
            class name in the specified folder. Default is 
            '~/dpeeg/out/model_name/'.
        maxEpochs_1, maxEpochs_2 : int
            Maximum number of epochs in the x stage of training. Default is
            1500 and 600 respectively.
        noIncreaseEpochs : int
            Maximum number of consecutive epochs when the accuracy of the first
            stage validation set has no relative improvement. Default is 100.
        varCheck : str
            The best value ('valInacc'/'valLoss') to check while determining 
            the best model which will be used for parameter initialization in
            the second stage of model training. Default is 'valInacc'.
        shuffle : bool
            Wheter shuffle dataset. Default is True.
        seed : int
            Seed of random for review. Default is DPEEG_SEED.
        verbose : int, str
            The log level of console. Default is INFO. Mainly used for debugging.

        All experimental results of the model will be saved in the outFolder
        directory of the trainer parameter.
        TODO: Nested Cross-Validation.
        '''
        self.trainer = trainer
        self.k = k
        self.maxEpochs_1 = maxEpochs_1
        self.maxEpochs_2 = maxEpochs_2
        self.noIncreaseEpochs = noIncreaseEpochs
        self.varCheck = varCheck

        self.loger = Logger('dpeeg_exp', clevel=verbose)
        self.timer = Timer()

        # set output folder
        netName = trainer.net.__class__.__name__
        self.outFolder = os.path.join(os.path.abspath(outFolder), netName) \
            if outFolder else os.path.join(DPEEG_DIR, 'out', netName)
        os.makedirs(self.outFolder, exist_ok=True)
        self.loger.info(f'Results will be saved in folder: {self.outFolder}')

        # create a stratified k-fold index
        self.skf = StratifiedKFold(k, shuffle=shuffle, random_state=seed)

    def kfold(
        self,
        trainset : Union[tuple, list],
        testset : Union[tuple, list],
        clsName : Union[tuple, list],
        sub : Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, dict]:
        '''Basic K-Fold cross-validation function.

        Parameters
        ----------
        trainset: tuple, list
            Dataset used for training module. If type is tuple or list, dataset
            should be (data, labels).
        testset:
            Dataset used for evaluating moduel. If type is tuple or list, dataset
            should be (data, labels).
        sub : str, optional
            Create a subdirectory of outFolder to store all yield results during 
            KFold_CV. If None, will use outFolder. Default is None.
        clsName : tuple, list
            The name of dataset labels.
        
        Returns
        -------
        Return testAcc, testKappa and results :
        
        results = {
            'expNo_x' : { ... },
            .
            .
            .
        }
        '''
        # save all exp results
        results = {}
        trainAccMetric, valAccMetric, testAccMetric = \
            MeanMetric(), MeanMetric(), MeanMetric()
        testPredsMetric, testTargetMetric = CatMetric(), CatMetric()

        # set kFold-cv results storage path
        subFolder = os.path.join(self.outFolder, sub) if sub else self.outFolder
        os.makedirs(subFolder, exist_ok=True)

        # store experiment results
        filer = Filer(os.path.join(subFolder, 'summary.txt'))
        cmFolder = os.path.join(subFolder, 'confusion_matrix')
        os.makedirs(cmFolder)

        # register a timer
        self.timer.start('kfold')

        # run k-fold
        for idx, (trainIdx, valIdx) in enumerate(self.skf.split(*trainset)):
            self.loger.info(f'\n# ------ {sub} - expNo.{idx+1} ------ #')

            # train one fold
            trainData = (trainset[0][trainIdx], trainset[1][trainIdx])
            valData = (trainset[0][valIdx], trainset[1][valIdx])
            expPath = os.path.join(subFolder, f'expNo_{idx + 1}')
            res = self.trainer.fit_with_val(
                trainData, valData, testset, expPath, clsName, self.maxEpochs_1,
                self.maxEpochs_2, self.noIncreaseEpochs, self.varCheck
            )

            # save all results
            results[f'expNo_{idx}'] = res
            trainRes, valRes, testRes = res['train'], res['val'], res['test']
            trainAccMetric.update(trainRes['acc'])
            valAccMetric.update(valRes['acc'])
            testAccMetric.update(testRes['acc'])
            testPredsMetric.update(testRes['preds'])
            testTargetMetric.update(testRes['target'])

            # save the confusion matirx image
            save_cm_img(testRes['preds'], testRes['target'], clsName,
                        os.path.join(cmFolder, f'expNo_{idx+1}_CM.png')
            )

            filer.write(f'------------------ expNo_{idx+1} ------------------\n')
            filer.write(f'Acc : Train={trainRes["acc"]:.4f} | Val={valRes["acc"]:.4f}' +
                        f' | Test={testRes["acc"]:.4f}\n\n')

        h, m, s = self.timer.stop('kfold')
        self.loger.info(f'\n[Cross-Validation Finish]')
        self.loger.info(f'Cost time = {h}H:{m}M:{s:.2f}S')

        # calculate the average acc
        trainAcc = trainAccMetric.compute()
        valAcc = valAccMetric.compute()
        testAcc = testAccMetric.compute()
        self.loger.info(f'Acc : train={trainAcc:.4f} | val={valAcc:.4f} | test={testAcc:.4f}') 
        
        # calculate cohen kappa
        testKappa = cohen_kappa(
            testPredsMetric.compute(), testTargetMetric.compute(),
            task = 'multiclass', num_classes = len(clsName)
        )

        # store the subject accuracy and cohen kappa on test set
        filer.write(f'--------- TestAvg ---------\n')
        filer.write(f'Acc = {testAcc*100:.2f}% | Kappa = {testKappa:.2f}\n')

        return testAcc, testKappa, results

    def run(
        self, 
        dataset : Union[EEGDataset, dict],
        clsName : Union[tuple, list], 
        datasetName : Optional[str] = None,
        desc : Optional[str] = None,
    ) -> dict:
        '''Run K-Fold cross validation on eeg datasets.

        Will select train set to process k-fold cross validation `kFold`, and 
        test set to evaluate the model.

        Parameters
        ----------
        dataset : EEGDataset, dict
            Dataset used for k-fold. Dataset structured as:
            
            {0 : { 'train': (array(nsample x channels x times), array(nsample)),
                   'test': (array(nsample x channels x times), array(nsample))},
             1 : {...},
            ...}
            
            where array can be `numpy.ndarray` or `torch.Tensor`. Please make 
            sure that the dataset has been split.
        clsName : tuple, list
            The name of dataset labels.
        datasetName : str, optional
            The dataset name to use. If None, `dataset.__class__.__name__` will 
            be used as the folder to save experimental results. If the type of 
            the dataset is dict, Please provide a dataset name to prevent 
            confusing results.
        desc : str, optional
            Add a short description to the current experiment. Default is None.

        Returns
        -------
        Return a dict of all subjects and corresponding experimental results.
        '''
        self.loger.info(f'\n[{self.k}-Fold Cross-Validation Starting]')

        # store all subjects' results in one folder
        self.outFolder = os.path.join(self.outFolder, dataset.__class__.__name__) \
            if not datasetName else os.path.join(self.outFolder, datasetName)
        os.makedirs(self.outFolder)
        filer = Filer(os.path.join(self.outFolder, 'summary.txt'))
        filer.write(f'[Start Time]: {self.timer.ctime()}\n\n')
        filer.write(f'[ExP Description]: {desc}\n\n')
        filer.write(str(self.trainer) + '\n\n')

        # save all sub results
        results = {}
        testAccMetric, testKappaMetric = MeanMetric(), MeanMetric()

        # update root timer and start k-fold cross validation for each subject
        self.timer.start()
        self.loger.info('=' * 50)
        for sub, data in dataset.items():

            self.loger.info(f'\n[Subject-{sub} Training ...]')
            self.loger.info(f'Train set = {data["train"][1].shape} | Test set = '
                  f'{data["test"][1].shape}')
            self.loger.info('-' * 50)

            # run a k-Fold cross validation
            testAcc, testKappa, subExpRes = \
                self.kfold(data['train'], data['test'], clsName, f'sub{sub}')

            results[f'sub_{sub}'] = subExpRes
            testAccMetric.update(testAcc)
            testKappaMetric.update(testKappa)

            filer.write(f'---------- Sub_{sub} ----------\n')
            filer.write(f'Acc = {testAcc*100:.2f}% | Kappa = {testKappa:.2f}\n\n')

        acc = testAccMetric.compute()
        kappa = testKappaMetric.compute()

        # store the model accuracy and cohen kappa
        filer.write(f'---------- MODEL ----------\n')
        filer.write(f'Acc = {acc*100:.2f}% | Kappa = {kappa:.2f}\n')
        with open(os.path.join(self.outFolder, f'results.pkl'), 'wb') as f:
            pickle.dump(results, f)

        # end
        h, m, s = self.timer.stop()
        self.loger.info(f'\n[All subjects finished]')
        self.loger.info(f'Cost time = {h}H:{m}M:{s:.2f}S')
        self.loger.info('=' * 50)
        self.loger.info(f'[Acc = {acc*100:.2f}% | Kappa = {kappa:.2f}]')

        return results


class HoldOut:
    '''TODO'''
    def __init__(self) -> None:
        pass
