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


import torch
import os, abc
from torch import Tensor
from typing import Literal
from sklearn.model_selection import StratifiedKFold
from torchmetrics.functional.classification.cohen_kappa import cohen_kappa
from torchmetrics.aggregation import MeanMetric, CatMetric

import dpeeg
from .classifier import TrainClassifier
from .metrics import AggMetrics
from ..data.datasets import EEGDataset
from ..data.functions import split_train_test, merge_train_test, check_dataset, check_sub_data
from ..data.transforms import Transforms
from ..tools import Timer, Filer, Logger
from ..utils import DPEEG_SEED, DPEEG_DIR, get_init_args
from .evaluate import save_cm_img


class Experiment(abc.ABC):
    '''Experiment base class.
    '''
    def __init__(
        self,
        trainer : TrainClassifier,
        out_folder : str | None = None,
        verbose : int | str = 'INFO'
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
        self.verbose = verbose

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
    def _run_sub(
        self,
        trainset : tuple | list,
        testset : tuple | list,
        cls_name : tuple | list,
        sub : str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
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
        Return test_acc, test_kappa, test_preds, test_target and results.
        '''
        pass

    def _trans_dataset(
        self, 
        trainset : list | tuple, 
        testset : list | tuple
    ) -> tuple:
        '''Apply pre-transforms on dataset.

        Generate a new virtual subject based on the input trainset and testset,
        then transform the new dataset and finally return the new trainset and
        testset.

        Parameters
        ----------
        trainset, testset : list, tuple
            List of data and labels.

        Returns
        -------
        Transformed trainset and testset.
        '''
        if self.transforms is not None:
            sub_dataset = {-1 : {'train':list(trainset), 'test':list(testset)}}
            sub_dataset = self.transforms(sub_dataset)
            trainset = sub_dataset[-1]['train']
            testset = sub_dataset[-1]['test']

        return trainset, testset
        
    def _process_sub_dataset(self, sub : int) -> tuple:
        '''Preprocess each subject's dataset.

        Different preprocessing operations are performed on the dataset accord-
        ing to different experimental requirement. By default, the training set
        and test set for each subject are returned.

        Parameters
        ----------
        sub : int
            Subjects currently undergoing the experiment.

        Returns
        -------
        trainset : tuple
            Tuple containing training data and labels.
        testset : tuple
            Tuple containing test data and labels.
        '''
        return self._trans_dataset(
            self.dataset[sub]['train'], self.dataset[sub]['test']
        )

    def run(
        self,
        dataset : EEGDataset | dict,
        cls_name : tuple | list,
        dataset_name : str | None = None,
        transforms : Transforms | None = None,
        desc : str | None = None,
    ) -> dict:
        '''Train models separately for each subject.

        This function will internally call the `_run_sub` function for each 
        subject, and save the final results together.

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
            the dataset is dict, Please provide a dataset name to prevent 
            confusing results.
        transforms : Transforms, optional
            Apply pre-transforms on dataset. Transformations will be apply 
            during the experiment on each subject's dataset. The rationable
            behind this method lies in deferring data manipulation, especially
            for certain transformations that could potentially enlarge the 
            dataset's memory footprint. This delay allows for the execution of 
            data manipulation after subject-independent experiment have concat-
            enated the relevant data (Time for Space) or the experiment subject
            are ready, mitigating the risk of memory overflow.
        desc : str, optional
            Add a short description to the current experiment.

        Returns
        -------
        Return a dict of all subjects and corresponding experimental results.
        '''
        if dataset_name:
            self.data_folder = os.path.join(self.out_folder, dataset_name)
        elif isinstance(dataset, EEGDataset):
            self.data_folder = \
                os.path.join(self.out_folder, dataset.__class__.__name__)
        else:
            self.data_folder = os.path.join(self.out_folder, 'Custom Dataset')
        os.makedirs(self.data_folder)

        filer = Filer(os.path.join(self.data_folder, 'summary.txt'))
        filer.write(f'[Start Time]: {self.timer.ctime()}\n')
        filer.write(f'[DPEEG Version]: {dpeeg.__version__}\n')
        filer.write(f'[Description]: {desc}\n')
        filer.write(str(self) + '\n')
        if isinstance(dataset, EEGDataset):
            filer.write(str(dataset) + '\n')
            dataset = dataset.dataset
        else:
            filer.write('[Custom dataset]\n')

        self.dataset = dataset
        self.transforms = transforms

        # save all sub results
        results = {}
        # acc_metric, kappa_metric = MeanMetric(), MeanMetric()
        acc_metric, kappa_metric = AggMetrics(), AggMetrics()
        preds_metric, target_metric = CatMetric(), CatMetric()

        # update root timer and start cross validation for each subject
        self.timer.start()
        self.loger.info('=' * 50)
        for sub in dataset.keys():
            self.loger.info(f'\n[Subject-{sub} Training ...]')
            trainset, testset = self._process_sub_dataset(sub)
            self.loger.info('-' * 50)

            # run one cross validation
            test_acc, test_kappa, test_preds, test_target, sub_exp_res = \
                self._run_sub(trainset, testset, cls_name, f'sub{sub}')

            results[f'sub_{sub}'] = sub_exp_res
            acc_metric.update(test_acc)
            kappa_metric.update(test_kappa)
            preds_metric.update(test_preds)
            target_metric.update(test_target)

            filer.write(f'---------- Sub_{sub} ----------\n')
            filer.write(f'Acc = {test_acc*100:.2f}% | Kappa = '
                        f'{test_kappa:.2f}\n\n')

        # acc = acc_metric.compute()
        # kappa = kappa_metric.compute()
        acc_mean = acc_metric.mean() * 100
        acc_std = acc_metric.std() * 100
        kappa_mean = kappa_metric.mean()

        filer.write(f'-------------- MODEL\n')
        # filer.write(f'Acc = {acc*100:.2f}% | Kappa = {kappa:.2f}\n')
        filer.write(f'Acc = {acc_mean:.2f}%\u00B1{acc_std:.2f} | Kappa = {kappa_mean:.2f}\n')
        save_cm_img(preds_metric.compute(), target_metric.compute(), 
                    cls_name, os.path.join(self.data_folder, 'CM.png'))
        torch.save(results, os.path.join(self.data_folder, f'results.pt'))

        h, m, s = self.timer.stop()
        self.loger.info(f'\n[All subjects finished]')
        self.loger.info(f'Cost time = {h}H:{m}M:{s:.2f}S')
        self.loger.info('=' * 50)
        self.loger.info(f'[Acc = {acc_mean:.2f}%\u00B1{acc_std:.2f} | Kappa = {kappa_mean:.2f}]')

        return results

    def __repr__(self) -> str:
        if self._repr:
            return self._repr
        else:
            class_name = self.__class__.__name__
            return f'{class_name} not implement attribute `self._repr`.'


class KFold(Experiment):
    def __init__(
        self,
        trainer : TrainClassifier,
        k : int = 5,
        out_folder : str | None = None,
        max_epochs_s1 : int = 1500,
        max_epochs_s2 : int = 600,
        no_increase_epochs : int = 100,
        second_stage : bool = True,
        load_best_state : bool = True,
        var_check : Literal['val_inacc', 'val_loss'] = 'val_inacc',
        isolate_testset : bool = True,
        shuffle : bool = True,
        seed : int = DPEEG_SEED,
        verbose : int | str = 'INFO',
    ) -> None:
        '''K-Fold cross validation experiment.

        Parameters
        ----------
        trainer : Train
            Trainer used for training module on dataset.
        k : int, optional
            k of k-Fold.
        out_folder : str, optional
            Store all experimental results in a folder named with the model 
            class name in the specified folder. Default is 
            '~/dpeeg/out/model_name/'.
        max_epochs_s1, max_epochs_s2 : int
            Maximum number of epochs in the x stage of training. Default is
            1500 and 600 respectively.
        no_increase_epochs : int
            Maximum number of consecutive epochs when the accuracy of the first
            stage validation set has no relative improvement.
        second_stage : bool
            If True, two-stage training will be performed.
        load_best_state : bool
            If True, stage 2 will retrain based on the best state in stage 1.
        var_check : str
            The best value ('val_inacc'/'val_loss') to check while determining 
            the best model which will be used for parameter initialization in
            the second stage of model training.
        isolate_testset : bool
            By default, the test set is independent, that is, the k-fold cross-
            validation at this time only divides the training set and the 
            verification set based on the training set to implement an early
            stopping mechanism, and finally evaluates on the isolated test set.
            If False, the test set is for each fold of k-fold cross-validation.
        shuffle : bool
            Shuffle before kfold.
        seed : int
            Seed of random for review.

        Notes
        -----
        If `isolate_testset` False, please provide the `transforms` parameter
        of the `run` function to avoid data leakage caused by operations such 
        as data augmentation in advance.

        TODO: Nested Cross-Validation.
        '''
        super().__init__(trainer, out_folder, verbose)

        self._repr = get_init_args(self, locals())
        self.max_epochs_s1 = max_epochs_s1
        self.max_epochs_s2 = max_epochs_s2
        self.no_increase_epochs = no_increase_epochs
        self.second_stage = second_stage
        self.load_best_state = load_best_state
        self.var_check = var_check
        self.isolate_testset = isolate_testset

        # create a stratified k-fold index
        self.skf = StratifiedKFold(k, shuffle=shuffle, random_state=seed)

    def _process_sub_dataset(self, sub: int) -> tuple:
        '''Whether to merge subject data depends on the experimental mode.
        '''
        if self.isolate_testset:
            return super()._process_sub_dataset(sub)

        if check_sub_data(sub, self.dataset[sub]):
            trainset = merge_train_test(*self.dataset[sub].values())
        else:
            trainset = self.dataset[sub]

        return trainset, None

    def _run_sub(
        self,
        trainset : tuple | list,
        testset : tuple | list,
        cls_name : tuple | list,
        sub : str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        '''Basic K-Fold cross validation function.

        Returns
        -------
        Return test_acc, test_kappa, test_preds, test_target and results :\n
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
        self.timer.start('kfold')

        # run k-fold
        for idx, (trainIdx, valIdx) in enumerate(self.skf.split(*trainset)):
            self.loger.info(f'\n# ------ {sub} - expNo.{idx+1} ------ #')

            if self.isolate_testset:
                train_set = (trainset[0][trainIdx], trainset[1][trainIdx])
                val_set = (trainset[0][valIdx], trainset[1][valIdx])
                test_set = testset
            else:
                train_set = [trainset[0][trainIdx], trainset[1][trainIdx]]
                test_set = [trainset[0][valIdx], trainset[1][valIdx]]
                train_set, test_set = self._trans_dataset(train_set, test_set)

                train_x, val_x, train_y, val_y = split_train_test(*train_set)
                train_set = (train_x, train_y)
                val_set = (val_x, val_y)

            # train one fold
            exp_path = os.path.join(sub_folder, f'expNo_{idx + 1}')
            res = self.trainer.fit_with_val(
                trainset=train_set,
                valset=val_set,
                testset=test_set,
                log_dir=exp_path,
                cls_name=cls_name,
                max_epochs_s1=self.max_epochs_s1,
                max_epochs_s2=self.max_epochs_s2,
                no_increase_epochs=self.no_increase_epochs,
                second_stage=self.second_stage,
                load_best_state=self.load_best_state,
                var_check=self.var_check # type: ignore
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
                        os.path.join(exp_path, f'{sub}_expNo_{idx+1}_CM.png'))

            filer.write(f'---------------- expNo_{idx+1} ----------------\n')
            filer.write(f'Acc : Train={train_res["acc"]:.4f}' +
                        f'|Val={val_res["acc"]:.4f}' +
                        f'|Test={test_res["acc"]:.4f}\n\n')

        h, m, s = self.timer.stop('kfold')
        self.loger.info(f'\n[Cross-Validation Finish]')
        self.loger.info(f'Cost time = {h}H:{m}M:{s:.2f}S')

        train_acc = train_acc_metric.compute()
        val_acc = val_acc_metric.compute()
        test_acc = test_acc_metric.compute()
        self.loger.info(f'Acc : train={train_acc:.4f} | val={val_acc:.4f} | ' +
                        f'test={test_acc:.4f}')

        test_preds = test_preds_metric.compute()
        test_target = test_target_metric.compute()
        save_cm_img(test_preds, test_target, cls_name,
                    os.path.join(sub_folder, f'{sub}_CM.png'))

        test_kappa = cohen_kappa(
            test_preds_metric.compute(), test_target_metric.compute(),
            task = 'multiclass', num_classes = len(cls_name)
        )

        # store the subject accuracy and cohen kappa on test set
        filer.write(f'--------- TestAvg ---------\n')
        filer.write(f'Acc = {test_acc*100:.2f}% | Kappa = {test_kappa:.2f}\n')

        return test_acc, test_kappa, test_preds, test_target, results


class Holdout(Experiment):
    def __init__(
        self,
        trainer : TrainClassifier,
        out_folder : str | None = None,
        max_epochs_s1 : int = 1500,
        no_increase_epochs : int = 200,
        var_check : Literal['train_inacc', 'train_loss', 'val_inacc', 'val_loss'] | None = None,
        split_val : bool = True,
        test_size : float = 0.25,
        second_stage : bool = True,
        load_best_state : bool = True,
        max_epochs_s2 : int = 600,
        seed : int = DPEEG_SEED,
        verbose : int | str = 'INFO'
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
            effect, representing the maximum number of epochs in `train_without
            _val`. 
            Default is 1500 and 600 respectively.
        no_increase_epochs : int
            Maximum number of consecutive epochs when the accuracy or loss of 
            the training set (or val set if split_val is True) has no relative 
            improvement. If set to the same as max_epochs_s1, early stopping 
            will not be performed.
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
            based on the results of the training set. If False, ignore 
            parameters test_size, max_epochs_s2, second_stage and seed.
        test_size : float
            The proportion of the validation set.
        second_stage : bool
            If True, two-stage training will be performed.
        load_best_state : bool
            If True, stage 2 will retrain based on the best state in stage 1 
            when second_stage is True or the best model parameters will be used
            for evaluation when second_stage is False.
        seed : int
            Seed of random for review.
        '''
        super().__init__(trainer, out_folder, verbose)

        self._repr = get_init_args(self, locals())
        self.max_epochs_s1 = max_epochs_s1
        self.no_increase_epochs = no_increase_epochs
        if var_check:
            self.var_check = var_check
        else:
            self.var_check = 'val_inacc' if split_val else 'train_loss'
        self.split_val = split_val
        self.test_size = test_size
        self.second_stage = second_stage
        self.load_best_state = load_best_state
        self.max_epochs_s2 = max_epochs_s2
        self.seed = seed

    def _run_sub(
        self,
        trainset : tuple | list,
        testset : tuple | list,
        cls_name : tuple | list,
        sub : str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
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
                trainset=(trainX, trainy),
                valset=(valX, valy),
                testset=testset,
                log_dir=sub_folder,
                cls_name=cls_name,
                max_epochs_s1=self.max_epochs_s1,
                max_epochs_s2=self.max_epochs_s2,
                no_increase_epochs=self.no_increase_epochs,
                second_stage=self.second_stage,
                load_best_state=self.load_best_state,
                var_check=self.var_check # type: ignore
            )
        else:
            result = self.trainer.fit_without_val(
                trainset=trainset,
                testset=testset,
                log_dir=sub_folder,
                cls_name=cls_name,
                max_epochs=self.max_epochs_s1,
                no_increase_epochs=self.no_increase_epochs,
                var_check=self.var_check, # type: ignore
                load_best_state=self.load_best_state
            )

        # save all results
        train_res, test_res = result['train'], result['test']
        train_acc, test_acc = train_res['acc'], test_res['acc']
        test_preds, test_target = test_res['preds'], test_res['target']
        save_cm_img(test_preds, test_target, cls_name, 
                    os.path.join(sub_folder, f'{sub}_CM.png')
        )
        self.loger.info(f'Acc : train={train_acc:.4f} | test={test_acc:.4f}')
        test_kappa = cohen_kappa(test_preds, test_target, task='multiclass',
                                 num_classes=len(cls_name))

        return test_acc, test_kappa, test_preds, test_target, result


class LOSO(Experiment):
    '''Leave-One-Subject-Out base class.

    In each fold of LOSO-CV, a single subject was used as the testing set, and
    the remaining N - 1 subjects were employed as the training set to obtain N 
    classification results. The training set was constructed using all data 
    sessions of all N - 1 training subjects for each dataset.
    '''
    def _merge_sub_dataset(self, exc_sub : int):
        '''Merge subject data except `sub`.
        '''
        merge_dataset = {}
        for sub, data in self.dataset.items():
            if sub != exc_sub:
                self.loger.debug(f'Merge sub_{sub} train and test set.')
                merge_dataset[sub] = merge_train_test(*data.values())

        self.loger.debug('Merge all subject dataset.')
        merge_dataset = merge_train_test(*merge_dataset.values())
        return merge_dataset

    def _process_sub_dataset(self, sub : int) -> tuple:
        '''Merge data sets according to LOSO experiment requirements.
        '''
        self.loger.info(f'[Leave Subject {sub} Out]')
        testset = merge_train_test(*self.dataset[sub].values())
        trainset = self._merge_sub_dataset(sub)
        trainset, testset = self._trans_dataset(trainset, testset)

        return trainset, testset


class LOSO_HO(LOSO, Holdout):
    '''Leave-One-Subject-Out Holdout cross validation experiment.
    '''
    pass


class LOSO_CV(LOSO, KFold):
    '''Leave-One-Subject-Out K-Fold cross validation experiment.
    '''
    def _process_sub_dataset(self, sub : int) -> tuple:
        if self.isolate_testset == False:
            raise  RuntimeError('LOSO_CV only support isolate testset.')
        return super()._process_sub_dataset(sub)