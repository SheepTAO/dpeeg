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

import dpeeg
from .classification import Train
from ..data.datasets import EEGDataset
from ..data.functions import split_train_test
from ..tools import Timer, Filer, Logger
from ..utils import DPEEG_SEED, DPEEG_DIR, get_init_args
from .evaluate import save_cm_img


class Experiment(abc.ABC):
    '''Experiment base class.
    '''
    def __init__(
        self,
        trainer : Train,
        out_folder : Optional[str] = None,
        verbose : Union[int, str] = 'INFO'
    ) -> None:
        '''Initialize the basic parameters of the experiment.

        Parameters
        ----------
        trainer : Train
            Trainer used for training module on dataset.
        out_folder : str, optional
            Store all experimental results in a folder named with the model 
            class name in the specified folder. Default is 
            '~/dpeeg/out/exp_name/model_name/'.
        verbose : int, str
            The log level of console. Default is INFO. Mainly used for debugg-
            ing.

        Notes
        -----
        The training results of all models for each subject will be saved under
        the out_folder directory.
        '''
        self._repr = None
        self.trainer = trainer

        # create loger and timer
        self.loger = Logger('dpeeg_exp', clevel=verbose)
        self.timer = Timer()

        # set output folder
        net = trainer.net.__class__.__name__
        exp = self.__class__.__name__
        self.out_folder = os.path.join(os.path.abspath(out_folder), net, exp) \
            if out_folder else os.path.join(DPEEG_DIR, 'out', net, exp)
        os.makedirs(self.out_folder, exist_ok=True)
        self.loger.info(f'Results will be saved in folder: {self.out_folder}')

    @abc.abstractmethod
    def run_sub(
        self,
        trainset : Union[tuple, list],
        testset : Union[tuple, list],
        cls_name : Union[tuple, list],
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
            Create a subdirectory of out_folder to store all yield results dur-
            ing subject training.
        cls_name : tuple, list
            The name of dataset labels.

        Returns
        -------
        Return test_acc, test_kappa and results.
        '''
        pass

    def _prepare_dataset(self, dataset : Union[EEGDataset, dict]):
        '''Preprocess the data set.

        Different preprocessing operations are performed on the dataset accord-
        ing to different experimental requirements.

        Parameters
        ----------
        dataset : EEGDataset, dict
            The dataset that will be processed.

        Notes
        -----
        The operations used should be in-place.
        '''
        pass

    def run(
        self,
        dataset : Union[EEGDataset, dict],
        cls_name : Union[tuple, list],
        dataset_name : Optional[str] = None,
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
        cls_name : tuple, list
            The name of dataset labels.
        dataset_name : str, optional
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
        if isinstance(dataset, EEGDataset):
            self.data_folder = \
                os.path.join(self.out_folder, dataset.__class__.__name__)
        else:
            self.data_folder = os.path.join(self.out_folder, dataset_name) \
                if dataset_name else os.path.join(self.out_folder, 'Custom')
        os.makedirs(self.data_folder)

        filer = Filer(os.path.join(self.data_folder, 'summary.txt'))
        filer.write(f'[Start Time]: {self.timer.ctime()}\n')
        filer.write(f'[DPEEG Version]: {dpeeg.__version__}\n')
        filer.write(f'[Description]: {desc}\n')
        filer.write(str(self) + '\n')
        if isinstance(dataset, EEGDataset):
            filer.write(str(dataset) + '\n')
        else:
            filer.write('[Custom dataset]\n')

        # save all sub results
        results = {}
        test_acc_metric, test_kappa_metric = MeanMetric(), MeanMetric()

        # preprocess data according to experimental needs
        self._prepare_dataset(dataset)

        # update root timer and start k-fold cross validation for each subject
        self.timer.start()
        self.loger.info('=' * 50)
        for sub, data in dataset.items():

            self.loger.info(f'\n[Subject-{sub} Training ...]')
            self.loger.info(f'Train set = {data["train"][1].shape} | Test set'
                  f' = {data["test"][1].shape}')
            self.loger.info('-' * 50)

            # run a k-Fold cross validation
            test_acc, test_kappa, sub_exp_res = self.run_sub(
                data['train'], data['test'], cls_name, f'sub{sub}'
            )

            results[f'sub_{sub}'] = sub_exp_res
            test_acc_metric.update(test_acc)
            test_kappa_metric.update(test_kappa)

            filer.write(f'---------- Sub_{sub} ----------\n')
            filer.write(f'Acc = {test_acc*100:.2f}% | Kappa = '
                        f'{test_kappa:.2f}\n\n')

        acc = test_acc_metric.compute()
        kappa = test_kappa_metric.compute()

        # store the model accuracy and cohen kappa
        filer.write(f'---------- MODEL ----------\n')
        filer.write(f'Acc = {acc*100:.2f}% | Kappa = {kappa:.2f}\n')
        with open(os.path.join(self.data_folder, f'results.pkl'), 'wb') as f:
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
        out_folder : Optional[str] = None,
        max_epochs_s1 : int = 1500,
        max_epochs_s2 : int = 600,
        no_increase_epochs : int = 100,
        second_stage : bool = True,
        var_check : Literal['val_inacc', 'val_loss'] = 'val_inacc',
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
        out_folder : str, optional
            Store all experimental results in a folder named with the model 
            class name in the specified folder. Default is 
            '~/dpeeg/out/model_name/'.
        max_epochs_s1, max_epochs_s2 : int
            Maximum number of epochs in the x stage of training. Default is
            1500 and 600 respectively.
        no_increase_epochs : int
            Maximum number of consecutive epochs when the accuracy of the first
            stage validation set has no relative improvement. Default is 100.
        second_stage : bool
            If True, two-stage training will be performed. Default is True.
        var_check : str
            The best value ('val_inacc'/'val_loss') to check while determining 
            the best model which will be used for parameter initialization in
            the second stage of model training. Default is 'val_inacc'.
        shuffle : bool
            Shuffle before kfold. Default is True.
        seed : int
            Seed of random for review. Default is DPEEG_SEED.

        TODO: Nested Cross-Validation.
        '''
        super().__init__(trainer, out_folder, verbose)

        self._repr = get_init_args(KFold, locals())
        self.max_epochs_s1 = max_epochs_s1
        self.max_epochs_s2 = max_epochs_s2
        self.no_increase_epochs = no_increase_epochs
        self.second_stage = second_stage
        self.var_check = var_check

        # create a stratified k-fold index
        self.skf = StratifiedKFold(k, shuffle=shuffle, random_state=seed)

    def run_sub(
        self,
        trainset : Union[tuple, list],
        testset : Union[tuple, list],
        cls_name : Union[tuple, list],
        sub : str,
    ) -> Tuple[Tensor, Tensor, dict]:
        '''Basic K-Fold cross validation function.

        Returns
        -------
        Return test_acc, test_kappa and results :\n
        results = {
            'expNo_1' : { ... },
            'expNo_2' : { ... },
            .
            .
        }
        '''
        # save all exp results
        results = {}
        train_acc_metric, val_acc_metric, test_acc_metric = \
            MeanMetric(), MeanMetric(), MeanMetric()
        test_preds_metric, test_target_metric = CatMetric(), CatMetric()

        # set subject's reults storage path
        sub_folder = os.path.join(self.data_folder, sub)
        os.makedirs(sub_folder)

        # store experiment results
        filer = Filer(os.path.join(sub_folder, 'summary.txt'))
        cmFolder = os.path.join(sub_folder, 'confusion_matrix')
        os.makedirs(cmFolder)

        # register a timer
        self.timer.start('kfold')

        # run k-fold
        for idx, (trainIdx, valIdx) in enumerate(self.skf.split(*trainset)):
            self.loger.info(f'\n# ------ {sub} - expNo.{idx+1} ------ #')

            # train one fold
            train_data = (trainset[0][trainIdx], trainset[1][trainIdx])
            val_data = (trainset[0][valIdx], trainset[1][valIdx])
            exp_path = os.path.join(sub_folder, f'expNo_{idx + 1}')
            res = self.trainer.fit_with_val(
                train_data, val_data, testset, exp_path, cls_name, 
                self.max_epochs_s1, self.max_epochs_s2, self.no_increase_epochs, 
                self.second_stage, var_check=self.var_check
            )

            # save all results
            results[f'expNo_{idx}'] = res
            train_res, val_res, test_res = res['train'], res['val'], res['test']
            train_acc_metric.update(train_res['acc'])
            val_acc_metric.update(val_res['acc'])
            test_acc_metric.update(test_res['acc'])
            test_preds_metric.update(test_res['preds'])
            test_target_metric.update(test_res['target'])

            # save the confusion matirx image
            save_cm_img(test_res['preds'], test_res['target'], cls_name,
                        os.path.join(cmFolder, f'{sub}_expNo_{idx+1}_CM.png')
            )

            filer.write(f'---------------- expNo_{idx+1} ----------------\n')
            filer.write(f'Acc : Train={train_res["acc"]:.4f}' +
                        f'|Val={val_res["acc"]:.4f}' +
                        f'|Test={test_res["acc"]:.4f}\n\n')

        h, m, s = self.timer.stop('kfold')
        self.loger.info(f'\n[Cross-Validation Finish]')
        self.loger.info(f'Cost time = {h}H:{m}M:{s:.2f}S')

        # calculate the average acc
        train_acc = train_acc_metric.compute()
        val_acc = val_acc_metric.compute()
        test_acc = test_acc_metric.compute()
        self.loger.info(f'Acc : train={train_acc:.4f} | val={val_acc:.4f} | ' +
                        f'test={test_acc:.4f}')

        # calculate cohen kappa
        test_kappa = cohen_kappa(
            test_preds_metric.compute(), test_target_metric.compute(),
            task = 'multiclass', num_classes = len(cls_name)
        )

        # store the subject accuracy and cohen kappa on test set
        filer.write(f'--------- TestAvg ---------\n')
        filer.write(f'Acc = {test_acc*100:.2f}% | Kappa = {test_kappa:.2f}\n')

        return test_acc, test_kappa, results


class Holdout(Experiment):
    def __init__(
        self,
        trainer : Train,
        out_folder : Optional[str] = None,
        max_epochs_s1 : int = 1500,
        no_increase_epochs : int = 200,
        var_check : Optional[
            Literal['train_inacc', 'train_loss', 'val_inacc', 'val_loss']
        ] = None,
        split_val : bool = True,
        test_size : float = 0.25,
        second_stage : bool = True,
        max_epochs_s2 : int = 600,
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
        out_folder : str, optional
            Store all experimental results in a folder named with the model 
            class name in the specified folder. Default is 
            '~/dpeeg/out/model_name/'.
        max_epochs_s1, max_epochs_s2 : int
            If split_val is True, max_epochs_s1 and max_epochs_s2 correspond to 
            the maximum number of epochs for the two stages of training in 
            `train_with_val`, respectively. Otherwise, only max_epochs_s1 takes 
            effect, representing the maximum number of epochs in `train_without_val`. 
            Default is 1500 and 600 respectively.
        no_increase_epochs : int
            Maximum number of consecutive epochs when the accuracy or loss of 
            the training set (or val set if split_val is True) has no relative 
            improvement. Default is 200. If set to the same as max_epochs_s1, 
            early stopping will not be performed.
        var_check : str
            The best value (train_inacc/train_loss if split_val is False; val_
            inacc/val_loss if split_val is True) to check while determining the 
            best model which will be used to evaluate its performance on the 
            test set. Default is 'val_inacc' (if split_val is True) or 'train_
            loss' (if split_val is False).
        split_val : bool
            If True, the training set will be split into training set and 
            validation set, and early stopping will be performed based on the
            corresponding metrics. Otherwise, early stopping will be performed 
            based on the results of the training set. If False, ignore paramet-
            ers test_size, max_epochs_s2, second_stage, shuffle and seed.
        test_size : float
            The proportion of the validation set. Default is 0.25.
        second_stage : bool
            If True, two-stage training will be performed. Default is True.
        seed : int
            Seed of random for review. Default is DPEEG_SEED.
        '''
        super().__init__(trainer, out_folder, verbose)

        self._repr = get_init_args(Holdout, locals())
        self.max_epochs_s1 = max_epochs_s1
        self.no_increase_epochs = no_increase_epochs
        if var_check:
            self.var_check = var_check
        else:
            self.var_check = 'val_inacc' if split_val else 'train_loss'
        self.split_val = split_val
        self.test_size = test_size
        self.second_stage = second_stage
        self.max_epochs_s2 = max_epochs_s2
        self.seed = seed

    def run_sub(
        self,
        trainset : Union[tuple, list],
        testset : Union[tuple, list],
        cls_name : Union[tuple, list],
        sub : str,
    ) -> Tuple[Tensor, Tensor, dict]:
        '''Basic holdout cross validation function.
        '''
        # set subject's reults storage path
        sub_folder = os.path.join(self.data_folder, sub)
        os.makedirs(sub_folder)

        # run hold-out
        self.loger.info(f'\n# ------------ {sub} ------------ #')
        if self.split_val:
            trainX, valX, trainy, valy = split_train_test(
                *trainset, test_size=self.test_size, seed=self.seed
            )
            result = self.trainer.fit_with_val(
                (trainX, trainy), (valX, valy), testset, sub_folder, cls_name,
                self.max_epochs_s1, self.max_epochs_s2, self.no_increase_epochs,
                self.second_stage, self.var_check
            )
        else:
            result = self.trainer.fit_without_val(
                trainset, testset, sub_folder, cls_name, self.max_epochs_s1, 
                self.no_increase_epochs, self.var_check
            )

        # save all results
        train_res, test_res = result['train'], result['test']
        train_acc, test_acc = train_res['acc'], test_res['acc']
        save_cm_img(test_res['preds'], test_res['target'], cls_name, 
                    os.path.join(sub_folder, f'{sub}_CM.png')
        )
        self.loger.info(f'Acc : train={train_acc:.4f} | test={test_acc:.4f}')
        test_kappa = cohen_kappa(test_res['preds'], test_res['target'], 
                                task='multiclass', num_classes=len(cls_name))

        return test_acc, test_kappa, result
