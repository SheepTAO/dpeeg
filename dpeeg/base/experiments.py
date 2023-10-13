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


import os, pickle, abc
from torch import Tensor
from typing import Union, Optional, Tuple
from typing_extensions import Literal
from sklearn.model_selection import StratifiedKFold
from torchmetrics.functional.classification.cohen_kappa import cohen_kappa
from torchmetrics.aggregation import MeanMetric, CatMetric

from .train import Train
from ..data.datasets import EEGDataset
from ..data.functions import split_train_test
from ..tools import Timer, Filer, Logger
from ..utils import DPEEG_SEED, DPEEG_DIR, get_class_init_args
from .evaluate import save_cm_img


class Experiment(abc.ABC):
    '''Experiment base class.
    '''
    def __init__(
        self,
        trainer : Train,
        outFolder : Optional[str] = None,
        verbose : Union[int, str] = 'INFO'
    ) -> None:
        '''Initialize the basic parameters of the experiment.

        Parameters
        ----------
        trainer : Train
            Trainer used for training module on dataset.
        outFolder : str, optional
            Store all experimental results in a folder named with the model 
            class name in the specified folder. Default is 
            '~/dpeeg/out/exp_name/model_name/'.
        verbose : int, str
            The log level of console. Default is INFO. Mainly used for debugg-
            ing.

        Notes
        -----
        The training results of all models for each subject will be saved under
        the outFolder directory.
        '''
        self._repr = None
        self.trainer = trainer

        # create loger and timer
        self.loger = Logger('dpeeg_exp', clevel=verbose)
        self.timer = Timer()

        # set output folder
        n = trainer.net.__class__.__name__
        e = self.__class__.__name__
        self.outFolder = os.path.join(os.path.abspath(outFolder), n, e) \
            if outFolder else os.path.join(DPEEG_DIR, 'out', n, e)
        os.makedirs(self.outFolder, exist_ok=True)
        self.loger.info(f'Results will be saved in folder: {self.outFolder}')

    @abc.abstractmethod
    def run_sub(
        self,
        trainset : Union[tuple, list],
        testset : Union[tuple, list],
        clsName : Union[tuple, list],
        sub : str,
    ) -> Tuple[Tensor, Tensor, dict]:
        '''Train a model on the specified subject data.

        This function will be called by `run` function to conduct experiments
        on the data of each individual subject. Reconstruct the model training 
        process according to different experimental requirements.

        Parameters
        ----------
        trainset : tuple, list
            Dataset used for training module. If type is tuple or list, dataset
            should be (data, labels).
        testset : tuple, list
            Dataset used for evaluating moduel. If type is tuple or list, data-
            set should be (data, labels).
        sub : str, optional
            Create a subdirectory of outFolder to store all yield results dur-
            ing subject training.
        clsName : tuple, list
            The name of dataset labels.

        Returns
        -------
        Return testAcc, testKappa and results.
        '''
        pass

    def run(
        self,
        dataset : Union[EEGDataset, dict],
        clsName : Union[tuple, list],
        datasetName : Optional[str] = None,
        desc : Optional[str] = None,
    ) -> dict:
        '''Train models separately for each subject.

        This function will internally call the `run_sub` function for each sub-
        ject, and save the final results together.

        Parameters
        ----------
        dataset : EEGDataset, dict
            Dataset structured as:\n
            {0 : { 'train': (array(nsample x channels x times), array(nsample)),
                   'test': (array(nsample x channels x times), array(nsample))},
             1 : {...},
            ...}\n
            where array can be `numpy.ndarray` or `torch.Tensor`. Please make 
            sure that the dataset has been split.
        clsName : tuple, list
            The name of dataset labels.
        datasetName : str, optional
            The dataset name to use. If None, `dataset.__class__.__name__` will 
            be used as the folder to save experimental results. If the type of 
            the dataset is dict, Please provide a dataset name to prevent conf-
            using results.
        desc : str, optional
            Add a short description to the current experiment. Default is None.

        Returns
        -------
        Return a dict of all subjects and corresponding experimental results.
        '''
        # store all subjects' results in one folder
        self.dataFolder = \
            os.path.join(self.outFolder, dataset.__class__.__name__) \
            if not datasetName else os.path.join(self.outFolder, datasetName)
        os.makedirs(self.dataFolder)
        filer = Filer(os.path.join(self.dataFolder, 'summary.txt'))
        filer.write(f'[Start Time]: {self.timer.ctime()}\n\n')
        filer.write(f'[Description]: {desc}\n\n')
        filer.write(str(self) + '\n\n')
        filer.write(str(dataset) + '\n\n')

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
                self.run_sub(data['train'], data['test'], clsName, f'sub{sub}')

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
        with open(os.path.join(self.dataFolder, f'results.pkl'), 'wb') as f:
            pickle.dump(results, f)

        # end
        h, m, s = self.timer.stop()
        self.loger.info(f'\n[All subjects finished]')
        self.loger.info(f'Cost time = {h}H:{m}M:{s:.2f}S')
        self.loger.info('=' * 50)
        self.loger.info(f'[Acc = {acc*100:.2f}% | Kappa = {kappa:.2f}]')

        return results

    def __repr__(self) -> str:
        if self._repr:
            return self._repr
        else:
            raise NotImplementedError(f'{self.__class__.__name__} not implement'
                                      ' attribute `self._repr`.')


class KFold(Experiment):
    def __init__(
        self,
        trainer : Train,
        k : int = 10,
        outFolder : Optional[str] = None,
        maxEpochs_1 : int = 1500,
        maxEpochs_2 : int = 600,
        noIncreaseEpochs : int = 100,
        secondStage : bool = True,
        varCheck : Literal['valInacc', 'valLoss'] = 'valInacc',
        shuffle : bool = True,
        seed : int = DPEEG_SEED,
        verbose : Union[int, str] = 'INFO',
    ) -> None:
        '''K-Fold cross validation experiment.

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
        secondStage : bool
            If True, two-stage training will be performed. Default is True.
        varCheck : str
            The best value ('valInacc'/'valLoss') to check while determining 
            the best model which will be used for parameter initialization in
            the second stage of model training. Default is 'valInacc'.
        shuffle : bool
            Shuffle before kfold. Default is True.
        seed : int
            Seed of random for review. Default is DPEEG_SEED.

        TODO: Nested Cross-Validation.
        '''
        super().__init__(trainer, outFolder, verbose)

        self._repr = get_class_init_args(KFold, locals())
        self.maxEpochs_1 = maxEpochs_1
        self.maxEpochs_2 = maxEpochs_2
        self.noIncreaseEpochs = noIncreaseEpochs
        self.secondStage = secondStage
        self.varCheck = varCheck

        # create a stratified k-fold index
        self.skf = StratifiedKFold(k, shuffle=shuffle, random_state=seed)

    def run_sub(
        self,
        trainset : Union[tuple, list],
        testset : Union[tuple, list],
        clsName : Union[tuple, list],
        sub : str,
    ) -> Tuple[Tensor, Tensor, dict]:
        '''Basic K-Fold cross validation function.

        Returns
        -------
        Return testAcc, testKappa and results :\n
        results = {
            'expNo_1' : { ... },
            'expNo_2' : { ... },
            .
            .
        }
        '''
        # save all exp results
        results = {}
        trainAccMetric, valAccMetric, testAccMetric = \
            MeanMetric(), MeanMetric(), MeanMetric()
        testPredsMetric, testTargetMetric = CatMetric(), CatMetric()

        # set subject's reults storage path
        subFolder = os.path.join(self.dataFolder, sub)
        os.makedirs(subFolder)

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
                self.maxEpochs_2, self.noIncreaseEpochs, self.secondStage, 
                varCheck=self.varCheck
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
                        os.path.join(cmFolder, f'{sub}_expNo_{idx+1}_CM.png')
            )

            filer.write(f'---------------- expNo_{idx+1} ----------------\n')
            filer.write(f'Acc : Train={trainRes["acc"]:.4f}' +
                        f'|Val={valRes["acc"]:.4f}' +
                        f'|Test={testRes["acc"]:.4f}\n\n')

        h, m, s = self.timer.stop('kfold')
        self.loger.info(f'\n[Cross-Validation Finish]')
        self.loger.info(f'Cost time = {h}H:{m}M:{s:.2f}S')

        # calculate the average acc
        trainAcc = trainAccMetric.compute()
        valAcc = valAccMetric.compute()
        testAcc = testAccMetric.compute()
        self.loger.info(f'Acc : train={trainAcc:.4f} | val={valAcc:.4f} | ' +
                        f'test={testAcc:.4f}')

        # calculate cohen kappa
        testKappa = cohen_kappa(
            testPredsMetric.compute(), testTargetMetric.compute(),
            task = 'multiclass', num_classes = len(clsName)
        )

        # store the subject accuracy and cohen kappa on test set
        filer.write(f'--------- TestAvg ---------\n')
        filer.write(f'Acc = {testAcc*100:.2f}% | Kappa = {testKappa:.2f}\n')

        return testAcc, testKappa, results


class Holdout(Experiment):
    def __init__(
        self,
        trainer : Train,
        outFolder : Optional[str] = None,
        maxEpochs_1 : int = 1500,
        noIncreaseEpochs : int = 200,
        varCheck : Optional[
            Literal['trainInacc', 'trainLoss', 'valInacc', 'valLoss']
        ] = None,
        splitVal : bool = True,
        testSize : float = 0.25,
        secondStage : bool = True,
        maxEpochs_2 : int = 600,
        seed : int = DPEEG_SEED,
        verbose : Union[int, str] = 'INFO'
    ) -> None:
        '''Holdout cross validation experiment.

        Validate the model using holdout cross validation, supports two model 
        training methods - only training set and training set split into valid-
        ation set.

        Parameters
        ----------
        trainer : Train
            Trainer used for training module on dataset.
        outFolder : str, optional
            Store all experimental results in a folder named with the model 
            class name in the specified folder. Default is 
            '~/dpeeg/out/model_name/'.
        maxEpochs_1, maxEpochs_2 : int
            If splitVal is True, maxEpochs_1 and maxEpochs_2 correspond to the
            maximum number of epochs for the two stages of training in `train_
            with_val`, respectively. Otherwise, only maxEpochs_1 takes effect, 
            representing the maximum number of epochs in `train_without_val`. 
            Default is 1500 and 600 respectively.
        noIncreaseEpochs : int
            Maximum number of consecutive epochs when the accuracy or loss of 
            the training set (or val set if splitVal is True) has no relative 
            improvement. Default is 200. If set to the same as maxEpochs_1, 
            early stopping will not be performed.
        varCheck : str
            The best value (trainInacc/trainLoss if splitVal is True; valInacc/
            valLoss if splitVal is False) to check while determining the best 
            model which will be used to evaluate its performance on the test 
            set. Default is 'valInacc' (if splitVal is True) or 'trainLoss' (if
            splitVal is False).
        splitVal : bool
            If True, the training set will be split into training set and 
            validation set, and early stopping will be performed based on the
            corresponding metrics. Otherwise, early stopping will be performed 
            based on the results of the training set. If False, ignore paramet-
            ers testSize, maxEpochs_2, secondStage, shuffle and seed.
        testSize : float
            The proportion of the validation set. Default is 0.25.
        secondStage : bool
            If True, two-stage training will be performed. Default is True.
        seed : int
            Seed of random for review. Default is DPEEG_SEED.
        '''
        super().__init__(trainer, outFolder, verbose)

        self._repr = get_class_init_args(Holdout, locals())
        self.maxEpochs_1 = maxEpochs_1
        self.noIncreaseEpochs = noIncreaseEpochs
        if varCheck:
            self.varCheck = varCheck
        else:
            self.varCheck = 'valInacc' if splitVal else 'trainLoss'
        self.splitVal = splitVal
        self.testSize = testSize
        self.secondStage = secondStage
        self.maxEpochs_2 = maxEpochs_2
        self.seed = seed

    def run_sub(
        self,
        trainset : Union[tuple, list],
        testset : Union[tuple, list],
        clsName : Union[tuple, list],
        sub : str,
    ) -> Tuple[Tensor, Tensor, dict]:
        '''Basic holdout cross validation function.
        '''
        # set subject's reults storage path
        subFolder = os.path.join(self.dataFolder, sub)
        os.makedirs(subFolder)

        # run hold-out
        self.loger.info(f'\n# ------------ {sub} ------------ #')
        if self.splitVal:
            trainX, valX, trainy, valy = split_train_test(
                *trainset, testSize=self.testSize, seed=self.seed
            )
            result = self.trainer.fit_with_val(
                (trainX, trainy), (valX, valy), testset, subFolder, clsName,
                self.maxEpochs_1, self.maxEpochs_2, self.noIncreaseEpochs,
                self.secondStage, self.varCheck
            )
        else:
            result = self.trainer.fit_without_val(
                trainset, testset, subFolder, clsName, self.maxEpochs_1, 
                self.noIncreaseEpochs, self.varCheck
            )

        # save all results
        trainRes, testRes = result['train'], result['test']
        trainAcc, testAcc = trainRes['acc'], testRes['acc']
        save_cm_img(testRes['preds'], testRes['target'], clsName, 
                    os.path.join(subFolder, f'{sub}_CM.png')
        )
        self.loger.info(f'Acc : train={trainAcc:.4f} | test={testAcc:.4f}')
        testKappa = cohen_kappa(testRes['preds'], testRes['target'], 
                                task='multiclass', num_classes=len(clsName))

        return testAcc, testKappa, result
