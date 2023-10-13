#!/usr/bin/env python
# coding: utf-8

"""
    The train model for any deep learning analysis.
    This class should provide following functionalities for any deep learning.

    module:
        1. train() -> Train the model
        2. predict() -> Evaluate the train, validation, and test performance
        3. Create train and validation graphs
        4. Run over CPU / GPU (if available)
    This class needs following things to run:
        1. net -> The architecture of the network. It should inherit nn.Module
             and should define the forward method.
        2. trainData, testData and valData -> these should be tuple, list or
             DataLoder the first parameters is `data` and second is `label`.
        3. optimizer -> the optimizer of type torch.optim.
        4. lr_scheduler -> the scheduler of type torch.optim.lr_scheduler.
        5. nGPU -> will run on GPU only if it's available, use CPU when the GPU
                is not available. 

    NOTE: The train model only support single-card training.

    @Author  : SheepTAO
    @Time    : 2023-07-24
"""


import os, torch, pickle
from torch import nn
from torch import optim
from torch import Tensor
from copy import deepcopy
from torchinfo import summary
from typing import Optional, Tuple, Union, Dict
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.aggregation import MeanMetric, CatMetric

from ..tools import Logger, Timer
from ..utils import DPEEG_SEED
from ..data.functions import to_tensor
from .stopcriteria import ComposeStopCriteria

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


# base train Model
class Train:
    '''Apex Trainer for any deep learning.
    '''
    def __init__(
        self,
        net : nn.Module,
        nGPU : int = 0,
        seed : int = DPEEG_SEED,
        lossFn : Union[str, nn.Module] = 'NLLLoss',
        lossFnArgs : Optional[dict] = None,
        optimizer : str = 'AdamW',
        optimArgs : Optional[dict] = None,
        lr : float = 1e-3,
        lrSch : Optional[str] = None,
        lrSchArgs : Optional[dict] = None,
        gradAcc : int = 1,
        batchSize : int = 32,
        dataSize : Optional[Union[tuple, list]] = None,
        depth : int = 3,
        verbose : Union[int, str] = 'INFO',
    ) -> None:
        '''Initialize the basic attribute of the train model.
        Generate a trainer to test the performance of the same network on different
        datasets.

        Parameters
        ----------
        net : nn.Module
            Inherit nn.Module and should define the forward method.
        nGPU : int
            Select the gpu id to train. Default is 0.
        seed : int
            Select random seed for review. Default is DPEEG_SEED.
        lossFn : str, nn.Module
            Name of the loss function from torch.nn which will be used for
            training. If Module, means using a custom loss function. Default is
            NLLLoss.
        lossFnArgs : dict, optional
            Additional arguments to be passed to the loss function. Default is 
            None.
        optimizer : str
            Name of the optimization function from torch.optim which will be
            used for training. Default is AdamW.
        optimArgs : dict, optional
            Additional arguments to be passed to the optimization function.
            Default is None.
        lr : float
            Learning rate. Default is 1e-3.
        lrSch : str, optional
            Name of the lr_scheduler from torch.optim.lr_scheduler which will 
            be used for training. Default is None.
        lrSchArgs : dict, optional
            Additional arguments to be passed to the lr_scheduler function.
            Default is None.
        gradAcc : int
            Aradient accumulation. Default is 1.
        batchSize : int
            Mini-batch size. Default is 32.
        dataSize : tuple, list, optional
            Output the structure of the network model according to the input
            dimension if the `dataSize` is given. Default is None.
        depth : int
            Depth of nested layers to display. Default is 3.
        verbose : int, str
            The log level of console. Default is INFO. Mainly used for debugg-
            ing.
        '''
        self.net = net

        self.loger = Logger('dpeeg_train', clevel=verbose)
        self.timer = Timer()

        # init trainer
        self.device = self.get_device(nGPU)
        self.net.to(self.device)
        self.set_seed(seed)

        # summarize network structure
        self.netArch = str(net) + '\n'
        if dataSize:
            self.netArch += str(summary(net, dataSize, depth=depth))
        self.loger.info(self.netArch)

        # the type of optimizer, etc. selected
        self.lossFnType = lossFn
        self.optimizerType = optimizer
        self.lrSchType = lrSch

        # save additional parameters
        self.lossFnArgs = lossFnArgs if lossFnArgs else {}
        self.optimArgs = optimArgs if optimArgs else {}
        self.lrSchArgs = lrSchArgs if lrSchArgs else {}
        # --- others
        self.lr = lr
        self.seed = seed
        self.gradAcc = gradAcc
        self.batchSize = batchSize
        self.verbose = verbose

        # set experimental details
        self.expDetails = {'expParam': {'seed': seed, 'lossFn': lossFn, 
            'lossFnArgs': lossFnArgs, 'optimizer': optimizer, 'lr': lr,
            'optimArgs': optimArgs, 'lrSch': lrSch, 'lrSchArgs': lrSchArgs,
            'batchSize': batchSize, 'gradAcc': gradAcc},
            'origNetParam': deepcopy(self.net.state_dict()),
        }

    def _run_one_epoch(
        self,
        trainLoader : DataLoader
    ) -> None:
        '''Run one epoch to train net.

        Parameters
        ----------
        trainLoader : DataLoader
            DataLoader used for training.
        '''
        # set the network in training mode
        self.net.train()

        # iterater over all the data
        with torch.enable_grad():
            for idx, (data, label) in enumerate(trainLoader):
                data, label = data.to(self.device), label.to(self.device)
                out = self.net(data)
                loss = self.lossFn(out, label)
                loss.backward()
                # gradient accumulation
                if((idx + 1) % self.gradAcc == 0):
                    # 1 - update parameters
                    self.optimizer.step()
                    # 2 - zero the parameter gradients
                    self.optimizer.zero_grad()
            # update lr
            # Note: Learning rate scheduling should be applied after optimizer’s update
            if self.lrSch:
                self.lrSch.step()

    def _predict(
        self,
        dataLoader : DataLoader,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        '''Predict the class of the input data.

        Parameters
        ----------
        dataLoader : DataLoader
            Dataset used for prediction.

        Returns
        -------
        preds : Tensor
            Predicted labels, as returned by a classifier.
        target : Tensor
            Ground truth (correct) labels.
        loss : Tensor
            Average loss.
        '''
        # set the network in the eval mode
        self.net.eval()

        lossSum = MeanMetric()
        preds, target = CatMetric(), CatMetric()

        # iterate over all the data
        with torch.no_grad():
            for data, label in dataLoader:
                data, label = data.to(self.device), label.to(self.device)
                out = self.net(data)
                loss = self.lossFn(out, label)
                lossSum.update(loss.item(), data.size(0))
                # convert the output of soft-max to class label
                # save preds and actual label
                out = out[0] if isinstance(out, tuple) else out
                preds.update(torch.argmax(out, dim=1).detach().cpu())
                target.update(label.cpu())
        return preds.compute(), target.compute(), lossSum.compute()

    def reset_fitter(
        self,
        logDir : str
    ) -> Tuple[str, SummaryWriter, Logger]:
        '''Reset the relevant parameters of the fitter.

        Reset the model's training parameters, learning rate schedule and opti-
        mizer etc. to their initialized state.

        Parameters
        ----------
        logDir : str
            Directory location (support hierarchical folder structure) to save
            training log.

        Returns
        -------
        Return the absolute file path and a new SummaryWriter object and logger
        manager for the fitter.
        '''
        # reset parameters of nn.Moudle and lr_scheduler
        self.net.load_state_dict(self.expDetails['origNetParam'])

        # create loss function
        self.lossFn = getattr(nn, self.lossFnType)(**self.lossFnArgs) \
            if isinstance(self.lossFnType, str) else self.lossFnType

        # create optimizer
        self.optimizer = getattr(optim, self.optimizerType) \
            (self.net.parameters(), lr=self.lr, **self.optimArgs)

        # create lr_scheduler
        self.lrSch = getattr(optim.lr_scheduler, self.lrSchType) \
            (self.optimizer, **self.lrSchArgs) if self.lrSchType else None

        # create log writer
        logDir = os.path.abspath(logDir)
        writer = SummaryWriter(logDir)
        loger = Logger(logDir, path=os.path.join(logDir, 'running.log'), 
                       flevel='INFO', clevel=self.verbose)
        return logDir, writer, loger

    def get_acc(self, preds : Tensor, target : Tensor, ncls : int) -> Tensor:
        '''Easy for program to caculate the accuarcy.
        '''
        return accuracy(preds, target, 'multiclass', num_classes=ncls)

    def set_seed(self, seed : int = DPEEG_SEED) -> None:
        '''Sets the seed for generating random numbers for cpu and gpu.
        '''
        torch.manual_seed(seed)
        if self.device != torch.device('cpu'):
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.loger.info(f'Set all random seed = {seed}')

    def get_device(self, nGPU : int = 0) -> torch.device:
        '''Get the device for training and testing.

        Parameters
        ----------
        nGPU : int
            GPU number to train on. Default is 0.
        '''
        if not torch.cuda.is_available():
            self.loger.info('GPU is not avaiable and the CPU will be used')
            dev = torch.device('cpu')
        else:
            if nGPU > torch.cuda.device_count() - 1:
                raise ValueError(f'GPU: {nGPU} does not exit.')
            dev = torch.device(f'cuda:{nGPU}')
            self.loger.info(f'Network will be trained in "cuda:{nGPU} ' +
                            f'({torch.cuda.get_device_name(dev)})"')
        return dev

    def _data_loader(
        self, 
        *datasets,
        batchSize : int = 32,
    ) -> DataLoader:
        '''Wrap multiple sets of data and labels and return DataLoader.

        Parameters
        ----------
        dataset : evey two sequence of indexables with same length / shape[0]
            Allowed inputs are lists and tuple of tensor or ndarray.
        batchSize : int
            Mini-batch size.
        '''
        nDataset = len(datasets)
        if nDataset % 2:
            raise ValueError('One dataset corresponds to one label set, but the '
                             f'total number of data got is {nDataset}.')
        if nDataset == 0:
            raise ValueError('At least one dataset required ad input.')

        # dataset wrapping tensors
        td = []
        for i in range(0, nDataset, 2):
            td.append(TensorDataset(*to_tensor(datasets[i], datasets[i + 1])))
        td = ConcatDataset(td)

        return DataLoader(td, batchSize, True, pin_memory=True)

    def fit_without_val(
        self,
        trainData : Union[tuple, list],
        testData : Union[tuple, list],
        logDir : str,
        clsName : Union[tuple, list],
        maxEpochs : int = 1000,
        noIncreaseEpochs : int = 200,
        varCheck : str = 'trainLoss',
    ) -> Dict[str, Dict[str, Tensor]]:
        '''During different training model processes, early stopping mechanisms
        can be executed using the training set (validation set not required) to
        select the model.

        Parameters
        ----------
        trainData : tuple, list
            Dataset used for training. If type is tuple or list, dataset should
            be (data, labels).
        valData : tuple, list
            Dataset used for validation. If type is tuple or list, dataset 
            should be (data, labels).
        testData : tuple, list
            Dataset used to evaluate the model. If type is tuple or list, 
            dataset should be (data, labels).
        logDir : str
            Directory location (support hierarchical folder structure) to save
            training log.
        clsName : tuple, list
            The name of dataset labels.
        maxEpochs : int
            Maximum number of epochs in training. Default is 1000.
        noIncreaseEpochs : int
            Maximum number of consecutive epochs when the accuracy or loss of 
            the training set has no relative improvement. Default is 200.
        varCheck : str
            The best value (trainInacc/trainLoss) to check while determining 
            the best model which will be used to evaluate its performance on 
            the test set. Default is 'trainLoss'.

        Returns
        -------
        Return train, validation and test results dict.
        {
            'train' : {'preds': Tensor, 'target': Tensor, 'acc': Tensor},
            'test'  : ...
        }
        '''
        logDir, writer, loger = self.reset_fitter(logDir)

        # check the best model
        bestVar = float('inf')
        bestNetParam = deepcopy(self.net.state_dict())

        checkList = ['trainInacc', 'trainLoss']
        if varCheck not in checkList:
            s = ', '.join(checkList)
            raise ValueError(f'Parameter `varCheck` only supports: {s}'
                             f' when training without val, but got {varCheck}')

        # initialize dataloader
        trainLoader = self._data_loader(*trainData, batchSize=self.batchSize)
        testLoader = self._data_loader(*testData, batchSize=self.batchSize)
        ncls = len(clsName)

        # start the training
        self.timer.start()
        loger.info(f'[Training...] - [{self.timer.ctime()}]')
        stopMon = ComposeStopCriteria({'Or': {
            'cri1' : {'MaxEpoch': {
                'maxEpochs': maxEpochs, 'varName': 'epoch'
            }},
            'cri2': {'NoDecrease': {
                'numEpochs': noIncreaseEpochs, 'varName': varCheck
            }}
        }})
        self.expDetails['fit'] = {
            'type': 'fit_without_val',
            'varCheck': varCheck,
            'stopMon': str(stopMon)
        }
        monitors = {'epoch': 0, 'trainLoss': float('inf'), 'trainInacc': 1}

        while not stopMon(monitors):

            # train one epoch
            self._run_one_epoch(trainLoader)

            # evaluate the training and validation accuracy
            trainPreds, trainTarget, trainLoss = self._predict(trainLoader)
            trainAcc = self.get_acc(trainPreds, trainTarget, ncls)
            monitors['trainInacc'] = 1 - trainAcc
            monitors['trainLoss'] = trainLoss

            # store loss and acc
            writer.add_scalars('train', {'loss': trainLoss, 'acc': trainAcc},
                               monitors['epoch'])
            loger.info(f'-->Epoch : {monitors["epoch"] + 1}')
            loger.info(f'  \u21b3train Loss/Acc = {trainLoss:.4f}/{trainAcc:.4f}')

            # select best model
            if monitors[varCheck] <= bestVar:
                bestVar = monitors[varCheck]
                bestNetParam = deepcopy(self.net.state_dict())

            monitors['epoch'] += 1

        writer.close()

        # report the checkpoint time of end and compute cost time
        loger.info(f'[Train Finish] - [{self.timer.ctime()}]')
        h, m, s = self.timer.stop()
        loger.info(f'Cost time = {h}H:{m}M:{s:.2f}S')

        # load the best model and evaulate this model in testData
        self.net.load_state_dict(bestNetParam)

        results = {}
        trainPreds, trainTarget, trainLoss = self._predict(trainLoader)
        trainAcc = self.get_acc(trainPreds, trainTarget, ncls)
        results['train'] = {
            'preds': trainPreds, 'target': trainTarget, 'acc': trainAcc
        }
        testPreds, testTarget, testLoss = self._predict(testLoader)
        testAcc = self.get_acc(testPreds, testTarget, ncls)
        results['test'] = {
            'preds': testPreds, 'target': testTarget, 'acc': testAcc
        }

        loger.info(f'Loss: train={trainLoss:.4f} | test={testLoss:.4f}')
        loger.info(f'Acc:  train={trainAcc:.4f} | test={testAcc:.4f}')

        # save the experiment details
        self.expDetails['results'] = results
        self.expDetails['bestNetParam'] = bestNetParam

        # store the training details
        with open(os.path.join(logDir, f'train.pkl'), 'wb') as f:
            pickle.dump(self.expDetails, f)

        # store the best net model parameters
        modelPath = os.path.join(logDir, f'train_checkpoint_best.pth')
        torch.save(bestNetParam, modelPath)

        return results

    def fit_with_val(
        self,
        trainData : Union[tuple, list],
        valData : Union[tuple, list],
        testData : Union[tuple, list],
        logDir : str,
        clsName : Union[tuple, list],
        maxEpochs_1 : int = 1500,
        maxEpochs_2 : int = 600,
        noIncreaseEpochs : int = 200,
        secondStage : bool = True,
        varCheck : str = 'valInacc',
    ) -> Dict[str, Dict[str, Tensor]]:
        '''Two-stage training strategy was used. In the first stage, the model 
        was trained using only the training set with the early stopping criteria
        whereby the validation set accuracy and loss was monitored and training
        was stopped if there was no increase in the validation set accuracy (or 
        loss) for consecutive 200 epochs. After reaching the stopping criteria, 
        network parameters with the best validation set accuracy (or loss) were
        restored. In the second stage, the model was trained with the complete 
        training data (train + validation set). The second stage training was
        stopped when the validation set loss reduced below the stage 1 training 
        set loss.

        Parameters
        ----------
        trainData : tuple, list
            Dataset used for training. If type is tuple or list, dataset should
            be (data, labels).
        valData : tuple, list
            Dataset used for validation. If type is tuple or list, dataset 
            should be (data, labels).
        testData : tuple, list
            Dataset used to evaluate the model. If type is tuple or list, 
            dataset should be (data, labels).
        logDir : str
            Directory location (support hierarchical folder structure) to save
            training log.
        clsName : tuple, list
            The name of dataset labels.
        maxEpochs_1, maxEpochs_2 : int
            Maximum number of epochs in the x stage of training. Default is
            1500 and 600 respectively.
        noIncreaseEpochs : int
            Maximum number of consecutive epochs when the accuracy or loss of 
            the first-stage validation set has no relative improvement. Default
            is 200.
        secondStage : bool
            If True, two-stage training will be performed. Default is True.
        varCheck : str
            The best value (valInacc/valLoss) to check while determining the 
            best model which will be used for parameter initialization in the
            second stage of model training. Default is 'valInacc'.

        Returns
        -------
        Return train, validation and test results dict.
        {
            'train' : {'preds': Tensor, 'target': Tensor, 'acc': Tensor},
            'test'  : ...,
            'val'   : ...
        }
        '''
        logDir, writer, loger = self.reset_fitter(logDir)

        # check the best model
        bestVar = float('inf')
        bestNetParam = deepcopy(self.net.state_dict())
        bestOptimParam = deepcopy(self.optimizer.state_dict())

        checkList = ['valInacc', 'valLoss']
        if varCheck not in checkList:
            s = ', '.join(checkList)
            raise ValueError(f'Parameter `varCheck` only supports: {s}'
                             f' when training with val, but got {varCheck}')

        # initialize dataloader
        trainLoader = self._data_loader(*trainData, batchSize=self.batchSize)
        valLoader = self._data_loader(*valData, batchSize=self.batchSize)
        testLoader = self._data_loader(*testData, batchSize=self.batchSize)
        ncls = len(clsName)

        # start the training
        self.timer.start()
        loger.info(f'[Training...] - [{self.timer.ctime()}]')

        stopMon = ComposeStopCriteria({'Or': {
            'cri1': {'MaxEpoch': {
                'maxEpochs': maxEpochs_1, 'varName': 'epoch'
            }},
            'cri2': {'NoDecrease': {
                'numEpochs': noIncreaseEpochs, 'varName': varCheck
            }}
        }})
        self.expDetails['fit'] = {
            'type': 'fit_with_val',
            'varCheck': varCheck,
            'stopMon_1': str(stopMon)
        }
        monitors = {'epoch': 0, 'valLoss': float('inf'), 'valInacc': 1,
                    'trainLoss': float('inf'), 'globalEpoch': 0}
        earlyStopReached, doStop = False, False

        while not doStop: 

            # train one epoch
            self._run_one_epoch(trainLoader)

            # evaluate the training and validation accuracy
            trainPreds, trainTarget, trainLoss = self._predict(trainLoader)
            trainAcc = self.get_acc(trainPreds, trainTarget, ncls)
            valPreds, valTarget, valLoss = self._predict(valLoader)
            valAcc = self.get_acc(valPreds, valTarget, ncls)
            monitors['valInacc'] = 1 - valAcc
            monitors['valLoss'] = valLoss

            # store loss and acc
            writer.add_scalars('train', {'loss': trainLoss, 'acc': trainAcc}, 
                                monitors['globalEpoch'])
            writer.add_scalars('val', {'loss': valLoss, 'acc': valAcc}, 
                                monitors['globalEpoch'])
            # print the epoch info
            loger.info(f'-->Epoch : {monitors["epoch"] + 1}')
            loger.info(f'  \u21b3train Loss/Acc = {trainLoss:.4f}/{trainAcc:.4f}'
                        f' | val Loss/Acc = {valLoss:.4f}/{valAcc:.4f}')

            # select best model on Stage 1
            if monitors[varCheck] <= bestVar:
                bestVar = monitors[varCheck]
                bestNetParam = deepcopy(self.net.state_dict())
                bestOptimParam = deepcopy(self.optimizer.state_dict())
                monitors['trainLoss'] = trainLoss

            monitors['epoch'] += 1
            monitors['globalEpoch'] += 1

            # check if to stop training
            if stopMon(monitors):
                # check whether to enter the second stage of training
                if secondStage and not earlyStopReached:
                    earlyStopReached = True
                    loger.info('[Early Stopping Reached] -> Continue '
                                    'training on full set.')
                    # load the best state
                    self.net.load_state_dict(bestNetParam)
                    self.optimizer.load_state_dict(bestOptimParam)

                    # Combine the train and val dataset
                    trainLoader = self._data_loader(*trainData, *valData,
                                                    batchSize=self.batchSize)

                    # update stop monitor and epoch
                    stopMon = ComposeStopCriteria({'Or': {
                        'cri1': {'MaxEpoch': {
                            'maxEpochs': maxEpochs_2, 'varName': 'epoch'
                        }},
                        'cri2': {'Smaller': {
                            'var': monitors['trainLoss'], 'varName': 'valLoss'
                        }}
                    }})
                    self.expDetails['fit']['stopMon_2'] = str(stopMon)
                    monitors['epoch'] = 0
                elif secondStage and earlyStopReached:
                    bestNetParam = deepcopy(self.net.state_dict())
                    doStop = True
                else:
                    doStop = True

        writer.close()

        # report the checkpoint time of end and compute cost time
        loger.info(f'[Train Finish] - [{self.timer.ctime()}]')
        h, m, s = self.timer.stop()
        loger.info(f'Cost Time = {h}H:{m}M:{s:.2f}S')

        # load the best model and evaulate this model in testData
        self.net.load_state_dict(bestNetParam)

        results = {}
        trainPreds, trainTarget, trainLoss = self._predict(trainLoader)
        trainAcc = self.get_acc(trainPreds, trainTarget, ncls)
        results['train'] = {
            'preds': trainPreds, 'target': trainTarget, 'acc': trainAcc
        }
        valPreds, valTarget, valLoss = self._predict(valLoader)
        valAcc = self.get_acc(valPreds, valTarget, ncls)
        results['val'] = {
            'preds': valPreds, 'target': valTarget, 'acc': valAcc
        }
        testPreds, testTarget, testLoss = self._predict(testLoader)
        testAcc = self.get_acc(testPreds, testTarget, ncls)
        results['test'] = {
            'preds': testPreds, 'target': testTarget, 'acc': testAcc
        }

        loger.info(f'Loss: train={trainLoss:.4f} | val={valLoss:.4f} | '
                   f'test={testLoss:.4f}')
        loger.info(f'Acc:  train={trainAcc:.4f} | val={valAcc:.4f} | '
                   f'test={testAcc:.4f}')

        # save the experiment details
        self.expDetails['results'] = results
        self.expDetails['bestNetParam'] = bestNetParam

        # store the training details
        with open(os.path.join(logDir, f'train.pkl'), 'wb') as f:
            pickle.dump(self.expDetails, f)

        # store the best net model parameters
        modelPath = os.path.join(logDir, f'train_checkpoint_best.pth')
        torch.save(bestNetParam, modelPath)

        return results

    def __repr__(self) -> str:
        '''Trainer details.
        '''
        s = '[Network architecture]:\n' + self.netArch + '\n'
        s += f'[Loss function]: {self.lossFnType}\n'
        if self.lossFnArgs:
            s += f'[LossFn Args]: {self.lrSchArgs}\n'
        s += f'[Optimizer]: {self.optimizerType}\n'
        s += f'[Learning rate]: {self.lr}\n'
        if self.optimArgs:
            s += f'[Optim Args]: {self.optimArgs}\n'
        if self.lrSchType:
            s += f'[Lr scheduler]: {self.lrSchType}\n'
            if self.lrSchArgs:
                s += f'[LrSch Args]: {self.lrSchArgs}\n'
        s += f'[Grad Acc]: {self.gradAcc}\n'
        s += f'[Batch Size]: {self.batchSize}\n'
        return s
