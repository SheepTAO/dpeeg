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


import os, torch, pickle, random
import numpy as np
from torch import nn
from torch import optim
from torch import Tensor
from copy import deepcopy
from torchinfo import summary
from ..tools import Logger, Timer
from typing import Optional, Tuple, Union, Dict
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from .stopcriteria import Criteria, ComposeStopCriteria, StrPath
from torchmetrics.functional.classification.accuracy import accuracy

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

DEEGDIR = os.path.join(os.path.expanduser('~'), 'deeg')
STOPCRI = {
    "cri_1": {
        "type": "MaxEpoch",
        "maxEpochs": 1000,
        "varName": "epoch"
    },
    
    "sym_2": "or",

    "cri_3": {
        "type": "NoDecrease",
        "numEpochs": 100,
        "varName": "valInacc"
    }
}


# base train Model
class Train:
    '''Apex Trainer for any deep learning.
    '''
    def __init__(
        self,
        net : nn.Module,
        classes : Union[list, tuple],
        nGPU : int = 0,
        stopCri : Optional[Union[Criteria, StrPath, dict]] = STOPCRI,
        seed : Optional[int] = None,
        lossFn : str = 'NLLLoss',
        lossFnArgs : dict = {},
        optimizer : str = 'AdamW',
        optimArgs : dict = {},
        lr : float = 1e-3,
        lrSch : Optional[str] = None,
        lrSchArgs : dict = {},
        gradAcc : int = 1,
        batchSize : int = 256,
        valCheck : str  = 'valInacc',
        outFolder : Optional[str] = None,
        overwrite : bool = False,
        dataSize : Optional[Union[tuple, list]] = None,
        depth : int = 3,
        pinMem : bool = True
    ) -> None:
        '''Initialize the basic attribute of the train model.

        Parameters
        ----------
        net : nn.Module
            Inherit nn.Module and should define the forward method.
        classes : list, tuple
            The name of given labels.
        nGPU : int, optional
            Select the gpu id to train. Default is 0.
        stopCri : Criteria, StrPath, dict, optional
            Criteria for training to stop. Default is max epochs = 1000, 
            no decrease epochs = 100.
        seed : int, optional
            Select random seed for review. Default is None.
        lossFn : str, optional
            Name of the loss function from torch.nn which will be used for
            training. Default is NLLLoss.
        lossFnArgs : dict, optional
            Additional arguments to be passed to the loss function. Default is {}.
        optimizer : str, optional
            Name of the optimization function from torch.optim which will be used
            for training. Default is AdamW.
        optimArgs : dict, optional
            Additional arguments to be passed to the optimization function.
            Default is {}.
        lr : float, optional
            Learning rate. Default is 1e-3.
        lrSch : str, optional
            Name of the lr_scheduler from torch.optim.lr_scheduler which will be used
            for training. Default is ''.
        lrSchArgs : dict, optional
            Additional arguments to be passed to the lr_scheduler function.
            Default is {}.
        gradAcc : int, optional
            Aradient accumulation. Default is 1.
        batchSize : int, optional
            Mini-batch size. Default is 256.
        valCheck : str, optional
            The best value ('valInacc'/'valLoss') to check while determining the best
            model. Default is 'valInacc'.
        outFolder : str, optional
            Store all results during training to the given folder. Default is
            '~/dpeeg/out/model_name/'.
        dataSize : tuple, list, optional
            Output the structure of the network model according to the input data
            dimension if the `dataSize` is given. Default is None.
        depth : int, optional
            Depth of nested layers to display. Default is 3.
        pinMem : bool, optional
            Whether to `pin_memory` to reduce data loading time. Default is True.
        '''
        self.net = net

        # set output folder
        if outFolder:
            self.outFolder = os.path.abspath(outFolder)
        else:
            self.outFolder = os.path.join(DEEGDIR, 'out', net.__class__.__name__)
        if not overwrite and os.path.exists(self.outFolder):
            raise FileExistsError(f'Results already exit in such folder: {self.outFolder}'
                                  ', result output overwrite is not allowed.')
        os.makedirs(self.outFolder, exist_ok=True)

        self.loger = Logger(f'_train_{net.__class__.__name__}', flevel=None)
        self.loger.info(f'Results will be save in folder: {self.outFolder}')

        # init trainer
        self.device = self.get_device(nGPU)
        self.net.to(self.device)
        
        if seed != None:
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
        self.valCheck = valCheck
        self.classes = classes
        self.pinMem = pinMem
        self.numClasses = len(classes)

        # set experimental details
        self.expDetails = {'seed': seed, 'expParam': {
            'lossFn': lossFn, 'lossFnArgs': lossFnArgs, 'optimizer': optimizer, 
            'optimArgs': optimArgs, 'lr': lr, 'lrSch': lrSch, 'lrSchArgs': lrSchArgs,
            'stopCir': str(stopCri), 'batchSize': batchSize}, 
            'origNetParam': deepcopy(self.net.state_dict()),
        }

        # set monitor of stop criteria
        if isinstance(stopCri, Criteria):
            self.stopMon = stopCri
        elif isinstance(stopCri, (StrPath, dict)):
            self.stopMon = ComposeStopCriteria(stopCri)
        else:
            raise TypeError(f'Cant parse \'{type(stopCri)}\'.')
        self.loger.info(f'Stop criteria: {str(self.stopMon)}')
    
    def run(
        self,
        trainData : Union[tuple, list, DataLoader],
        valData : Union[tuple, list, DataLoader],
        testData : Optional[Union[tuple, list, DataLoader]] = None,
        logDir : Optional[str] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        '''Apex function to train and test network.

        Parameters
        ----------
        trainData : tuple, list, DataLoader
            Dataset used for training. If type is tuple or list, dataset should
            be (data, labels).
        valData : tuple, list, DataLoader
            Dataset used for validation. If type is tuple or list, dataset should
            be (data, labels).
        testData : tuple, list, DataLoader, optional
            Dataset used to evaluate the model. Default is None.
        logDir : str, optional
            Save directory location (under outFolder) and support hierarchical folder 
            structure. Default is None, which means use outFolder.
        
        Returns
        -------
        According to the input dataset, return train, validation and test (if testData
        is given) results dict.
        {
            'train' : {'preds': Tensor, 'acts': Tensor, 'acc': Tensor},
            'val'   : ...,
            'test'  : ...(if testData is given)
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
        bestVal = float('inf')
        bestNetParam = deepcopy(self.net.state_dict())

        # create log writer
        logDir = os.path.join(self.outFolder, logDir) if logDir else self.outFolder
        writer = SummaryWriter(logDir)
        loger = Logger(logDir, path=os.path.join(logDir, 'running.log'), mode='a')
        
        # reset stop criteria
        self.stopMon.reset_variables()
        
        # initialize dataloader
        trainLoader = self._data_loader(trainData, self.batchSize, self.pinMem)
        valLoader = self._data_loader(valData, self.batchSize, self.pinMem)
        testLoader = self._data_loader(testData, self.batchSize, self.pinMem) \
            if testData else None
        
        # start the training
        monitors = {'epoch': 0, 'valLoss': float('inf'), 'valInacc': 1}
        doStop = False
        bestRes = {}
        timer = Timer()
        loger.info(f'[Training...] - [{timer.ctime()}]')
        
        while not doStop: 
            
            # train one epoch
            self._run_one_epoch(trainLoader)

            # evaluate the training and validation accuracy
            trainPreds, trainActs, trainLoss = self._predict(trainLoader)
            trainAcc = self.get_acc(trainPreds, trainActs)
            monitors['trainInacc'] = 1 - trainAcc
            monitors['trainLoss'] = trainLoss

            valPreds, valActs, valLoss = self._predict(valLoader)
            valAcc = self.get_acc(valPreds, valActs)
            monitors['valInacc'] = 1 - valAcc
            monitors['valLoss'] = valLoss

            # store loss and acc
            writer.add_scalars('loss', {'train': trainLoss, 'val': valLoss},
                                monitors['epoch'])
            writer.add_scalars('acc', {'train': trainAcc, 'val': valAcc}, 
                               monitors['epoch'])

            # print the epoch info
            loger.info(f'---Epoch : {monitors["epoch"] + 1}')
            loger.info(f'   train Loss/Acc = {trainLoss:.4f} / {trainAcc:.4f}' +
                       f' | val Loss/Acc = {valLoss:.4f} / {valAcc:.4f}')

            # select best model
            if monitors[self.valCheck] <= bestVal:
                bestVal = monitors[self.valCheck]
                bestNetParam = deepcopy(self.net.state_dict())
                bestRes['train'] = {
                    'preds': trainPreds, 'acts': trainActs, 'acc': trainAcc, 'loss': trainLoss
                }
                bestRes['test'] = {
                    'preds': valPreds, 'acts': valActs, 'acc': valAcc, 'loss': valLoss
                }
            
            # check if to stop
            if self.stopMon(monitors):
                doStop = True
            monitors['epoch'] += 1
        
        writer.close()

        # report the checkpoint time of end and compute cost time
        loger.info(f'[Train Finish] - [{timer.ctime()}]')
        h, m, s = timer.stop()
        loger.info(f'Cost Time = {h}H:{m}M:{s:.2f}S')

        # load the best result
        trainLoss, valLoss = bestRes["train"]["loss"], bestRes["test"]["loss"]
        printLoss = f'Loss: train={trainLoss:.4f} | val={valLoss:.4f}'
        trainAcc, valAcc = bestRes["train"]["acc"], bestRes["val"]["acc"]
        printAcc = f'Acc : train={trainAcc:.4f} | val={valAcc:.4f}'

        # save the experiment details
        self.expDetails['result'] = {'bestTrainAcc' : bestRes['train']['acc']}
        self.expDetails['result'] = {'bestValAcc' : bestRes['val']['acc']}
        self.expDetails['bestNetParam'] = bestNetParam
        
        # load the best model and evaulate this model in testData (if not None)
        if testLoader:
            self.net.load_state_dict(bestNetParam)
            testPreds, testActs, testLoss = self._predict(testLoader)
            testAcc = self.get_acc(testPreds, testActs)
            bestRes['test'] = {
                'preds': testPreds, 'acts': testActs, 'acc': testAcc, 'loss': testLoss
            }
            printLoss = printLoss+f' | test={testLoss:.4f}'
            printAcc = printAcc+f' | test={testAcc:.4f}'
            self.expDetails['result'] = {'bestTestAcc' : testAcc}
            
        # print the result of best model
        loger.info(printLoss)
        loger.info(printAcc)

        # store the experiment details
        with open(os.path.join(logDir, f'train.pkl'), 'wb') as f:
            pickle.dump(self.expDetails, f)

        # store the best net model parameters
        modelPath = os.path.join(logDir, f'train_checkpoint_best.pth')
        torch.save(bestNetParam, modelPath)
        
        return bestRes

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

    def set_seed(self, seed : int = 3407) -> None:
        '''Set all the random initializations with a given seed.
        '''
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device != torch.device('cpu'):
            torch.cuda.manual_seed_all(seed)

        self.loger.info(f'Set all random seed = {seed}')

    def get_device(self, nGPU : int = 0) -> torch.device:
        '''Get the device for training and testing.

        Parameters
        ----------
        nGPU : int, optional
            GPU number to train on. Default is 0.
        '''
        if not torch.cuda.is_available():
            self.loger.info('GPU is not avaiable and the CPU will be used')
            dev = torch.device('cpu')
        else:
            if nGPU > torch.cuda.device_count()-1:
                raise ValueError(f'GPU: {nGPU} does not exit.')
            dev = torch.device(f'cuda:{nGPU}')
            self.loger.info(f'Network will be trained in "cuda:{nGPU} ' +
                            f'({torch.cuda.get_device_name(dev)})"')
        
        return dev
    
    def _data_loader(
        self, 
        dataset : Union[tuple, list, DataLoader],
        batchSize : int = 256,
        pinMem : bool = True
    ) -> DataLoader:
        '''Wrap the data and labels and return DataLoader.

        Parameters
        ----------
        dataset : tuple/list(tensor/ndarray, tensor/ndarray), dataloader
            Return the corresponding `DataLoader` with according to the dataset.
        '''
        if isinstance(dataset, DataLoader):
            DataLoader.pin_memory = pinMem
            return dataset

        if len(dataset) != 2:
            raise ValueError('The first dimension of input must be 2 (data, label), ' +
                             f'but get {len(dataset)}.')
        
        if isinstance(dataset[0], np.ndarray):
            dataset = [torch.as_tensor(v) for v in dataset]
        dataset = (dataset[0].float(), dataset[1].long())
        
        return DataLoader(TensorDataset(*dataset), batch_size=batchSize,
                          shuffle=True, pin_memory=pinMem)

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
        s += f'[Stop Cri]:\t{str(self.stopMon)}\n'
        s += f'[Value check]:\t{self.valCheck}\n'
        s += f'[Classes name]:\t{self.classes}\n'
