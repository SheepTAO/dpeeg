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


import os, torch
from torch import Tensor
from .train import Train
from ..tools import Timer, Filer
from .evaluate import cal_cm_and_plt_img
from typing import Union, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from torchmetrics.functional.classification.cohen_kappa import cohen_kappa
from torch.utils.tensorboard.writer import SummaryWriter
from ..data.datasets import EEGDataset


class KFoldCV:
    def __init__(
        self,
        trainer : Train,
        k : int = 10,
        shuffle : bool = True,
        seed : int = 3407,
    ) -> None:
        '''K-Fold cross-validation class.

        Parameters
        ----------
        trainer : Train
            Trainer used for training module on dataset.
        k : int, optional
            k of k-Fold. Default is 10.
        shuffle : bool, optional
            Wheter shuffle dataset. Default is True.
        seed : int, optional
            Seed of random for review. Default is 3407.

        All experimental results of the model will be saved in the outFolder
        directory of the trainer parameter.
        TODO: Nested Cross-Validation.
        '''

        self.trainer = trainer
        self.k = k
        
        self.outFolder = trainer.outFolder
        self.classes = trainer.classes

        # create a stratified k-fold index
        self.skf = StratifiedKFold(k, shuffle=shuffle, random_state=seed)

        print(f'\n[{self.skf.get_n_splits()}-Fold Cross-Validation Init]')

    def kfold(
        self,
        trainset : Union[tuple, list],
        testset : Union[tuple, list],
        sub : Optional[str]
    ) -> Tuple[Tensor, ...]:
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
            KFold_CV. Default is None.
        
        Returns
        -------
        Return trainAcc, valAcc, testAcc and testKappa.
        '''
        def get_res(
            results : dict, 
            task : str, 
            var : str = 'acc'
        ) -> Tensor:
            '''Get all results from the results by var and return them.
            '''
            res = []
            for i in range(self.k):
                data = results[task][i][var].unsqueeze(0) if var == 'acc' \
                    else results[task][i][var]
                res.append(data)
            return torch.cat(res)

        # save trained results
        results = {
            'train' : [],
            'val'   : [],
            'test'  : [],
        }

        # set kFold-cv results storage path
        subFolder = os.path.join(self.outFolder, sub) if sub else self.outFolder
        os.makedirs(subFolder, exist_ok=True)

        # create a writer to store confusion_matrix
        cmFolder = os.path.join(subFolder, 'confusion_matrix')
        writer = SummaryWriter(cmFolder)

        # store all experiment results in one file
        resPath = os.path.join(subFolder, 'resutls.txt')
        filer = Filer(resPath)

        # create a timer to report cost time
        timer = Timer()
        timer.start()

        # run k-fold
        for idx, (trainIdx, valIdx) in enumerate(self.skf.split(*trainset)):
            print(f'\n# ------------- {sub} - expNo.{idx+1} ------------- #')

            # train one fold
            trainData = (trainset[0][trainIdx], trainset[1][trainIdx])
            valData = (trainset[0][valIdx], trainset[1][valIdx])
            expPath = os.path.join(subFolder, f'expNo_{idx + 1}')
            res = self.trainer.run(trainData, valData, testset, expPath)
            trainRes, valRes, testRes = res['train'], res['val'], res['test']
            
            # save the confusion matirx image
            writer.add_figure(
                'trainCM_with_exp',
                cal_cm_and_plt_img(trainRes['preds'], trainRes['acts'], self.classes),
                global_step = idx + 1
            )
            writer.add_figure(
                'valCM_with_exp',
                cal_cm_and_plt_img(valRes['preds'], valRes['acts'], self.classes),
                global_step = idx + 1
            )
            writer.add_figure(
                'testCM_with_exp',
                cal_cm_and_plt_img(testRes['preds'], testRes['acts'], 
                                   self.classes, store=True,
                                   imgPath=os.path.join(cmFolder, f'expNo_{idx+1}_CM.png')),
                global_step = idx + 1
            )

            # save all results
            results['train'].append(trainRes)
            results['val'].append(valRes)
            results['test'].append(testRes)

            filer.write(f'------------------ ExpNo_{idx+1} ------------------\n')
            filer.write(f'Acc : Train={trainRes["acc"]:.4f} | Val={valRes["acc"]:.4f}' +
                        f' | Test={testRes["acc"]:.4f}\n\n')
        
        h, m, s = timer.stop()
        print(f'\n[Cross-Validation Finish]')
        print(f'Cost time = {h}H:{m}M:{s:.2f}S')

        # calculate the acerage acc
        trainAccList = get_res(results, 'train')
        valAccList = get_res(results, 'val')
        testAccList = get_res(results, 'test')

        trainAcc = torch.mean(trainAccList)
        valAcc = torch.mean(valAccList)
        testAcc = torch.mean(testAccList)
        print(f'Acc : train={trainAcc:.4f} | val={valAcc:.4f} | test={testAcc:.4f}') 
        
        # calculate cohen kappa
        testKappa = cohen_kappa(
            get_res(results, 'test', 'preds'),
            get_res(results, 'test', 'acts'),
            task = 'multiclass',
            num_classes = len(self.classes)
        )

        # store the subject accuracy and cohen kappa
        filer.write(f'------------- Average -------------\n')
        filer.write(f'Acc = {testAcc*100:.2f}\u00b1{torch.std(testAccList)*100:.2f}% | '+
                    f'Kappa = {testKappa:.2f}\n')

        return trainAcc, valAcc, testAcc, testKappa

    def kFold_eeg(
        self, 
        dataset : Union[EEGDataset, dict],
        datasetName : Optional[str] = None,
        desc : Optional[str] = None
    ) -> None:
        '''K-Fold cross validation for eeg datasets.

        Will select train set to process k-fold cross validation `kFold`, and 
        test set to evaluate the model.

        Parameters
        ----------
        dataset : EEGDataset
            Dataset used for k-fold. Dataset structured as:
            
            {'1' : { 'train': (array(nsample x channels x times), array(nsample)),
                     'test': (array(nsample x channels x times), array(nsample))},
             '2' : {...},
            ...}
            
            where array can be `numpy.ndarray` or `torch.Tensor`. Please make sure 
            that the dataset has been split.
        datasetName : str, optional
            The dataset name to use. If None, `dataset.__class__.__name__` will be
            used to save experimental results.
        desc : str, optional
            Add a short description to the current experiment. Default is None.
        '''
        # store all subjects' results in one file
        dataPath = os.path.join(self.outFolder, dataset.__class__.__name__) \
            if not datasetName else os.path.join(self.outFolder, datasetName)
        resPath = os.path.join(dataPath, 'resutls.txt')
        filer = Filer(resPath)
        filer.write(f'[Start Time]: {Timer.ctime()}\n\n')
        if desc:
            filer.write(f'[ExP Description]: {desc}\n\n')
        filer.write(str(self.trainer) + '\n\n')
            
        # resport results
        testAccList = []
        testKappaList = []

        print('=' * 50)
        for sub, data in dataset.items():

            print(f'\n[Subject-{sub} Training ...]')
            print(f'Train set = {data["train"][1].shape} | Test set = '
                  f'{data["test"][1].shape}')
            print('-' * 50)

            # run a k-Fold cross validation
            trainAcc, valAcc, testAcc, testKappa = \
                self.kfold(trainset=data['train'], testset=data['test'], 
                           sub=os.path.join(dataPath, f'sub{sub}'))
            
            testAccList.append(testAcc.unsqueeze(0))
            testKappaList.append(testKappa.unsqueeze(0))

            filer.write(f'---------- Sub_{sub} ----------\n')
            filer.write(f'Acc = {testAcc*100:.2f}% | Kappa = {testKappa:.2f}\n\n')

        acc = torch.mean(torch.cat(testAccList))
        std = torch.std(torch.cat(testAccList))
        kappa = torch.mean(torch.cat(testKappaList))
        # store the model accuracy and cohen kappa
        filer.write(f'------------- MODEL ------------\n')
        filer.write(f'Acc = {acc*100:.2f}\u00b1{std*100:.2f}% | Kappa = {kappa:.2f}\n')

        # end
        print(f'\n[All subjects finished]')
        print('=' * 50)
        print(f'[Acc = {acc*100:.2f}\u00b1{std*100:.2f}% | Kappa = {kappa:.2f}]')
        

class HoldOut:
    '''TODO'''
    def __init__(self) -> None:
        pass
