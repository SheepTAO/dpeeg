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
        5. outFolder -> the folder where the results will be stored.
        6. nGPU -> will run on GPU only if it's available
             use CPU when the GPU is not available. 

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

from ..tools import Logger, Timer
from ..utils import DPEEG_SEED, DPEEG_DIR
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
        classes : Union[list, tuple],
        stopCri : Union[str, dict],
        nGPU : int = 0,
        seed : int = DPEEG_SEED,
        lossFn : str = 'NLLLoss',
        lossFnArgs : dict = {},
        optimizer : str = 'AdamW',
        optimArgs : dict = {},
        lr : float = 1e-3,
        lrSch : Optional[str] = None,
        lrSchArgs : dict = {},
        gradAcc : int = 1,
        batchSize : int = 256,
        varCheck : str  = 'valLoss',
        outFolder : Optional[str] = None,
        dataSize : Optional[Union[tuple, list]] = None,
        depth : int = 3,
        pinMem : bool = True
    ) -> None:
        '''Initialize the basic attribute of the train model.
        Generate a trainer to test the performance of the same network on different
        datasets.

        Parameters
        ----------
        net : nn.Module
            Inherit nn.Module and should define the forward method.
        classes : list, tuple
            The name of given labels.
        nGPU : int
            Select the gpu id to train. Default is 0.
        stopCri : str, dict
            Criteria for training to stop. Default is max epochs = 1000, val loss
            no decrease epochs = 100.
        seed : int
            Select random seed for review. Default is DPEEG_SEED.
        lossFn : str
            Name of the loss function from torch.nn which will be used for
            training. Default is NLLLoss.
        lossFnArgs : dict
            Additional arguments to be passed to the loss function. Default is {}.
        optimizer : str
            Name of the optimization function from torch.optim which will be used
            for training. Default is AdamW.
        optimArgs : dict
            Additional arguments to be passed to the optimization function.
            Default is {}.
        lr : float
            Learning rate. Default is 1e-3.
        lrSch : str
            Name of the lr_scheduler from torch.optim.lr_scheduler which will be 
            used for training. Default is ''.
        lrSchArgs : dict
            Additional arguments to be passed to the lr_scheduler function.
            Default is {}.
        gradAcc : int
            Aradient accumulation. Default is 1.
        batchSize : int
            Mini-batch size. Default is 256.
        varCheck : str
            The best value ('valInacc'/'valLoss') to check while determining 
            the best model. Default is 'valLoss'.
        outFolder : str, optional
            Store all results during training to the given folder. Default is
            '~/dpeeg/out/model_name/'.
        dataSize : tuple, list, optional
            Output the structure of the network model according to the input data
            dimension if the `dataSize` is given. Default is None.
        depth : int
            Depth of nested layers to display. Default is 3.
        pinMem : bool
            Whether to `pin_memory` to reduce data loading time. Default is True.
        '''
        self.net = net

        # set output folder
        if outFolder:
            self.outFolder = os.path.abspath(outFolder)
        else:
            self.outFolder = os.path.join(DPEEG_DIR, 'out', net.__class__.__name__)
        os.makedirs(self.outFolder, exist_ok=True)
        if os.listdir(self.outFolder):
            raise FileExistsError(f'{self.outFolder} is not a empty folder.')

        checkList = ['valInacc', 'valLoss']
        if varCheck not in checkList:
            raise ValueError(f'Parameter `valCheck` only supports: {checkList}, '
                             f'but got {varCheck}')

        self.loger = Logger('dpeeg_train', flevel=None)
        self.loger.info(f'Results will be saved in folder: {self.outFolder}')

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
        self.lossFnArgs = lossFnArgs
        self.optimArgs = optimArgs
        self.lrSchArgs = lrSchArgs
        # --- others
        self.lr = lr
        self.seed = seed
        self.gradAcc = gradAcc
        self.batchSize = batchSize
        self.stopCri = stopCri
        self.varCheck = varCheck
        self.classes = classes
        self.pinMem = pinMem
        self.numClasses = len(classes)

        # set experimental details
        self.expDetails = {'expParam': {'seed': seed, 'lossFn': lossFn, 
            'lossFnArgs': lossFnArgs, 'optimizer': optimizer, 'lr': lr,
            'optimArgs': optimArgs, 'lrSch': lrSch, 'lrSchArgs': lrSchArgs,
            'batchSize': batchSize, 'gradAcc': gradAcc, 'varCheck': varCheck},
            'origNetParam': deepcopy(self.net.state_dict()),
        }
    
    def run(
        self,
        trainData : Union[tuple, list, DataLoader],
        valData : Union[tuple, list, DataLoader],
        testData : Union[tuple, list, DataLoader],
        logDir : Optional[str] = None,
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
        trainData : tuple, list, DataLoader
            Dataset used for training. If type is tuple or list, dataset should
            be (data, labels).
        valData : tuple, list, DataLoader
            Dataset used for validation. If type is tuple or list, dataset should
            be (data, labels).
        testData : tuple, list, DataLoader
            Dataset used to evaluate the model. If type is tuple or list, dataset
            should be (data, labels).
        logDir : str, optional
            Save directory location (under outFolder) and support hierarchical folder 
            structure. Default is None, which means use outFolder.
        
        Returns
        -------
        According to the input dataset, return train, validation and test (if testData
        is given) results dict.
        {
            'train' : {'preds': Tensor, 'acts': Tensor, 'acc': Tensor, 'loss': Tensor},
            'test'  : ...
            'val'   : ...,
        }
        '''
        # reset parameters of nn.Moudle and lr_scheduler
        self.net.load_state_dict(self.expDetails['origNetParam'])

        # create loss function
        self.lossFn = getattr(nn, self.lossFnType)(**self.lossFnArgs)

        # create optimizer
        self.optimizer = getattr(optim, self.optimizerType) \
            (self.net.parameters(), lr=self.lr, **self.optimArgs)
        
        # create lr_scheduler
        self.lrSch = getattr(optim.lr_scheduler, self.lrSchType) \
            (self.optimizer, **self.lrSchArgs) if self.lrSchType else None

        # check the best model
        bestVar = float('inf')
        bestNetParam = deepcopy(self.net.state_dict())
        bestOptimParam = deepcopy(self.optimizer.state_dict())

        # create log writer
        logDir = os.path.join(self.outFolder, logDir) if logDir else self.outFolder
        writer = SummaryWriter(logDir)
        loger = Logger(logDir, path=os.path.join(logDir, 'running.log'), mode='a')
        
        # initialize dataloader
        trainLoader = self._data_loader(trainData, self.batchSize, self.pinMem)
        valLoader = self._data_loader(valData, self.batchSize, self.pinMem)
        testLoader = self._data_loader(testData, self.batchSize, self.pinMem)
        
        # start the training
        timer = Timer()
        loger.info(f'[Training...] - [{timer.ctime()}]')

        stopMon = ComposeStopCriteria({'And': {
            'cri1': {'MaxEpoch': {
                'maxEpochs': 1500, 'varName': 'epoch'
            }},
            'cri2': {'NoDecrease': {
                'numEpochs': 200, 'varName': 'valLoss'
            }}
        }})
        monitors = {'epoch': 0, 'valLoss': float('inf'), 'valInacc': 1}
        earlyStopReached, doStop = False, False
        
        while not doStop: 
            
            # train one epoch
            self._run_one_epoch(trainLoader)

            # evaluate the training and validation accuracy
            trainPreds, trainActs, trainLoss = self._predict(trainLoader)
            trainAcc = self.get_acc(trainPreds, trainActs)

            valPreds, valActs, valLoss = self._predict(valLoader)
            valAcc = self.get_acc(valPreds, valActs)
            monitors['valInacc'] = 1 - valAcc
            monitors['valLoss'] = valLoss

            # store loss and acc
            writer.add_scalars('train', {'loss': trainLoss, 'acc': trainAcc}, 
                                monitors['epoch'])
            writer.add_scalars('val', {'loss': valLoss, 'acc': valAcc}, 
                                monitors['epoch'])
            # print the epoch info
            loger.info(f'-->Epoch : {monitors["epoch"] + 1}')
            loger.info(f'   train Loss/Acc = {trainLoss:.4f}/{trainAcc:.4f}'
                        f' | val Loss/Acc = {valLoss:.4f}/{valAcc:.4f}')

            # select best model on Stage 1
            if monitors[self.varCheck] <= bestVar:
                bestVar = monitors[self.varCheck]
                bestNetParam = deepcopy(self.net.state_dict())
                bestOptimParam = deepcopy(self.optimizer.state_dict())
            
            # check if to stop training
            if stopMon(monitors):
                # check whether to enter the second stage of training
                if not earlyStopReached:
                    earlyStopReached = True
                    self.loger.info('[Early Stopping Reached] -> Start training '
                                    'on both training set and val set.')
                    # load the best state
                    self.net.load_state_dict(bestNetParam)
                    self.optimizer.load_state_dict(bestOptimParam)

                    # Combine the train and val dataset
                    trainLoader = self._data_loader(trainData, valData)

                    # update stop monitor and epoch
                    stopMon = ComposeStopCriteria({'Or': {
                        'cri1': {'MaxEpoch': {
                            'maxEpochs': 600, 'varName': 'epochs'
                        }},
                        'cir2': {'Smaller': {
                            'var': trainLoss, 'varName': 'valLoss'
                        }}
                    }})
                    monitors['epoch'] = 0
                else:
                    bestNetParam = deepcopy(self.net.state_dict())
                    doStop = True
            monitors['epoch'] += 1
        
        writer.close()

        # report the checkpoint time of end and compute cost time
        loger.info(f'[Train Finish] - [{timer.ctime()}]')
        h, m, s = timer.stop()
        loger.info(f'Cost Time = {h}H:{m}M:{s:.2f}S')

        # load the best model and evaulate this model in testData
        self.net.load_state_dict(bestNetParam)

        results = {}
        trainPreds, trainActs, trainLoss = self._predict(trainLoader)
        trainAcc = self.get_acc(trainPreds, trainActs)
        results['train'] = {
            'preds': trainPreds, 'acts': trainActs, 'acc': trainAcc, 'loss': trainLoss,
        }
        valPreds, valActs, valLoss = self._predict(valLoader)
        valAcc = self.get_acc(valPreds, valActs)
        results['val'] = {
            'preds': valPreds, 'acts': valActs, 'acc': valActs, 'loss': valLoss
        }
        testPreds, testActs, testLoss = self._predict(testLoader)
        testAcc = self.get_acc(testPreds, testActs)
        results['test'] = {
            'preds': testPreds, 'acts': testActs, 'acc': testAcc, 'loss': testLoss
        }

        loger.info(f'Loss: train={trainLoss:.4f} | val={valLoss:.4f} | '
                   f'test={testLoss:.4f}')
        loger.info(f'Acc: train={trainAcc:.4f} | val={valAcc:.4f} | '
                   f'test={testAcc:.4f}')

        # save the experiment details
        self.expDetails['result'] = results
        self.expDetails['bestNetParam'] = bestNetParam

        # store the training details
        with open(os.path.join(logDir, f'train.pkl'), 'wb') as f:
            pickle.dump(self.expDetails, f)

        # store the best net model parameters
        modelPath = os.path.join(logDir, f'train_checkpoint_best.pth')
        torch.save(bestNetParam, modelPath)
        
        return results

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
    ) -> Tuple[Tensor, Tensor, float]:
        '''Predict the class of the input data.

        Parameters
        ----------
        dataLoader : DataLoader
            Dataset used for prediction.

        Returns
        -------
        predicted : Tensor
            Tensor of predicted labels.
        actual : Tensor
            Tensor of actual labels.
        loss : float
            Average loss.
        '''
        predicted = torch.empty(0)
        actual = torch.empty(0)
        lossSum = 0

        # set the network in the eval mode
        self.net.eval()

        # iterate over all the data
        with torch.no_grad():
            for data, label in dataLoader:
                data, label = data.to(self.device), label.to(self.device)
                out = self.net(data)
                lossSum += self.lossFn(out, label).detach().item()
                # convert the output of soft-max to class label
                preds = torch.argmax(out, dim=1)
                # save preds and actual label
                predicted = torch.cat((predicted, preds.detach().cpu()))
                actual = torch.cat((actual, label.cpu()))
        
        return predicted, actual, lossSum / len(dataLoader)
    
    def get_acc(self, preds : Tensor, acts : Tensor) -> Tensor:
        '''Easy for program to caculate the accuarcy.
        '''
        return accuracy(preds, acts, 'multiclass', num_classes=self.numClasses)

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
        batchSize : int = 256,
        pinMem : bool = True
    ) -> DataLoader:
        '''Wrap multiple sets of data and labels and return DataLoader.

        Parameters
        ----------
        dataset : evey two sequence of indexables with same length / shape[0]
            Allowed inputs are lists and tuple of tensor or ndarray.
        '''
        nDataset = len(datasets)
        if nDataset % 2:
            raise ValueError('One data set corresponds to one label set, but the '
                             f'total number of data got is {nDataset}.')
        if nDataset == 0:
            raise ValueError('At least one dataset required ad input.')

        # dataset wrapping tensors
        td = []
        for i in range(0, nDataset, 2):
            td.append(TensorDataset(*to_tensor(datasets[i], datasets[i + 1])))
        td = ConcatDataset(td)

        return DataLoader(td, batchSize, True, pin_memory=pinMem)

    def __repr__(self) -> str:
        '''Trainer details.
        '''
        s = '[Network architecture]:\n' + self.netArch
        s += f'[Loss function]:\t{self.lossFnType}\n'
        if self.lossFnArgs:
            s += f'[LossFn Args]:\t{self.lrSchArgs}\n'
        s += f'[Optimizer]:\t{self.optimizerType}\n'
        s += f'[Learning rate]:\t{self.lr}\n'
        if self.optimArgs:
            s += f'[Optim Args]:\t{self.optimArgs}\n'
        if self.lrSchType:
            s += f'[Lr scheduler]:\t{self.lrSchType}\n'
            if self.lrSchArgs:
                s += f'[LrSch Args]:\t{self.lrSchArgs}\n'
        s += f'[Seed]:\t{self.seed}\n'
        s += f'[Grad Acc]:\t{self.gradAcc}\n'
        s += f'[Value check]:\t{self.varCheck}\n'
        s += f'[Classes name]:\t{self.classes}\n'
        return s
