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
        2. train_data, test_data and val_data -> these should be tuple, list or
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
import torch.nn as nn
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
        loss_fn : Union[str, nn.Module] = 'NLLLoss',
        loss_fn_args : Optional[dict] = None,
        optimizer : str = 'Adam',
        optimizer_args : Optional[dict] = None,
        lr : float = 1e-3,
        lr_sch : Optional[str] = None,
        lr_sch_args : Optional[dict] = None,
        grad_acc : int = 1,
        batch_size : int = 32,
        data_size : Optional[Union[tuple, list]] = None,
        depth : int = 3,
        verbose : Union[int, str] = 'INFO',
    ) -> None:
        '''Initialize the basic attribute of the train model.

        Generate a trainer to test the performance of the same network on 
        different datasets.

        Parameters
        ----------
        net : nn.Module
            Inherit nn.Module and should define the forward method.
        nGPU : int
            Select the gpu id to train. Default is 0.
        seed : int
            Select random seed for review. Default is DPEEG_SEED.
        loss_fn : str, nn.Module
            Name of the loss function from torch.nn which will be used for
            training. If Module, means using a custom loss function. Default is
            NLLLoss.
        loss_fn_args : dict, optional
            Additional arguments to be passed to the loss function. Default is 
            None.
        optimizer : str
            Name of the optimization function from torch.optim which will be
            used for training. Default is Adam.
        optimizer_args : dict, optional
            Additional arguments to be passed to the optimization function.
            Default is None.
        lr : float
            Learning rate. Default is 1e-3.
        lr_sch : str, optional
            Name of the lr_scheduler from torch.optim.lr_scheduler which will 
            be used for training. Default is None.
        lr_sch_args : dict, optional
            Additional arguments to be passed to the lr_scheduler function.
            Default is None.
        grad_acc : int
            Aradient accumulation. Default is 1.
        batch_size : int
            Mini-batch size. Default is 32.
        data_size : tuple, list, optional
            Output the structure of the network model according to the input
            dimension if the `data_size` is given. Default is None.
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
        self.net_arch = str(net) + '\n'
        if data_size:
            self.net_arch += str(summary(net, data_size, depth=depth))
        self.loger.info(self.net_arch)

        # the type of optimizer, etc. selected
        self.loss_fn_type = loss_fn
        self.optimizer_type = optimizer
        self.lr_sch_type = lr_sch

        # save additional parameters
        self.loss_fn_args = loss_fn_args if loss_fn_args else {}
        self.optimizer_args = optimizer_args if optimizer_args else {}
        self.lr_sch_args = lr_sch_args if lr_sch_args else {}
        # --- others
        self.lr = lr
        self.seed = seed
        self.grad_acc = grad_acc
        self.batch_size = batch_size
        self.verbose = verbose

        # set experimental details
        self.exp_details = {
            'expParam': {
                'seed': seed, 
                'loss_fn': loss_fn, 
                'loss_fn_args': loss_fn_args, 
                'optimizer': optimizer, 
                'optimizer_args': optimizer_args, 
                'lr': lr,
                'lr_sch': lr_sch, 
                'lr_sch_args': lr_sch_args, 
                'batch_size': batch_size, 
                'grad_acc': grad_acc
            }, 
            'orig_net_param': deepcopy(self.net.state_dict()),
        }

    def _run_one_epoch(
        self,
        train_loader : DataLoader
    ) -> None:
        '''Run one epoch to train net.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader used for training.
        '''
        # set the network in training mode
        self.net.train()

        # iterater over all the data
        with torch.enable_grad():
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.to(self.device), label.to(self.device)
                out = self.net(data)
                loss = self.loss_fn(out, label)
                loss.backward()
                # gradient accumulation
                if((idx + 1) % self.grad_acc == 0):
                    # 1 - update parameters
                    self.optimizer.step()
                    # 2 - zero the parameter gradients
                    self.optimizer.zero_grad()
            # update lr
            # Note: Learning rate scheduling should be applied after optimizer’s update
            if self.lr_sch:
                self.lr_sch.step()

    def _predict(
        self,
        data_loader : DataLoader,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        '''Predict the class of the input data.

        Parameters
        ----------
        data_loader : DataLoader
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

        loss_sum = MeanMetric()
        preds, target = CatMetric(), CatMetric()

        # iterate over all the data
        with torch.no_grad():
            for data, label in data_loader:
                data, label = data.to(self.device), label.to(self.device)
                out = self.net(data)
                loss = self.loss_fn(out, label)
                loss_sum.update(loss.item(), data.size(0))
                # convert the output of soft-max to class label
                # save preds and actual label
                out = out[0] if isinstance(out, tuple) else out
                preds.update(torch.argmax(out, dim=1).detach().cpu())
                target.update(label.cpu())
        return preds.compute(), target.compute(), loss_sum.compute()

    def reset_fitter(
        self,
        log_dir : str
    ) -> Tuple[str, SummaryWriter, Logger]:
        '''Reset the relevant parameters of the fitter.

        Reset the model's training parameters, learning rate schedule and opti-
        mizer etc. to their initialized state.

        Parameters
        ----------
        log_dir : str
            Directory location (support hierarchical folder structure) to save
            training log.

        Returns
        -------
        str, SummaryWriter, Logger:
            Return the absolute file path and a new SummaryWriter object and 
            logger manager for the fitter.
        '''
        # reset parameters of nn.Moudle and lr_scheduler
        self.net.load_state_dict(self.exp_details['orig_net_param'])

        # create loss function
        self.loss_fn = getattr(nn, self.loss_fn_type)(**self.loss_fn_args) \
            if isinstance(self.loss_fn_type, str) else self.loss_fn_type

        # create optimizer
        self.optimizer = getattr(optim, self.optimizer_type) \
            (self.net.parameters(), lr=self.lr, **self.optimizer_args)

        # create lr_scheduler
        self.lr_sch = getattr(optim.lr_scheduler, self.lr_sch_type) \
            (self.optimizer, **self.lr_sch_args) if self.lr_sch_type else None

        # create log writer
        log_dir = os.path.abspath(log_dir)
        writer = SummaryWriter(log_dir)
        loger = Logger(log_dir, path=os.path.join(log_dir, 'running.log'), 
                       flevel='INFO', clevel=self.verbose)
        return log_dir, writer, loger

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
        batch_size : int = 32,
    ) -> DataLoader:
        '''Wrap multiple sets of data and labels and return DataLoader.

        Parameters
        ----------
        dataset : evey two sequence of indexables with same length / shape[0]
            Allowed inputs are lists and tuple of tensor or ndarray.
        batch_size : int
            Mini-batch size.
        '''
        num_dataset = len(datasets)
        if num_dataset % 2:
            raise ValueError('One dataset corresponds to one label set, but the'
                             f' total number of data got is {num_dataset}.')
        if num_dataset == 0:
            raise ValueError('At least one dataset required as input.')

        # dataset wrapping tensors
        td = []
        for i in range(0, num_dataset, 2):
            td.append(TensorDataset(*to_tensor(datasets[i], datasets[i + 1])))
        td = ConcatDataset(td)

        return DataLoader(td, batch_size, True, pin_memory=True)

    def fit_without_val(
        self,
        train_data : Union[tuple, list],
        test_data : Union[tuple, list],
        log_dir : str,
        cls_name : Union[tuple, list],
        max_epochs : int = 1000,
        no_increase_epochs : int = 200,
        var_check : str = 'train_loss',
    ) -> Dict[str, Dict[str, Tensor]]:
        '''During different training model processes, early stopping mechanisms
        can be executed using the training set (validation set not required) to
        select the model.

        Parameters
        ----------
        train_data : tuple, list
            Dataset used for training. If type is tuple or list, dataset should
            be (data, labels).
        test_data : tuple, list
            Dataset used to evaluate the model. If type is tuple or list, 
            dataset should be (data, labels).
        log_dir : str
            Directory location (support hierarchical folder structure) to save
            training log.
        cls_name : tuple, list
            The name of dataset labels.
        max_epochs : int
            Maximum number of epochs in training. Default is 1000.
        no_increase_epochs : int
            Maximum number of consecutive epochs when the accuracy or loss of 
            the training set has no relative improvement. Default is 200.
        var_check : str
            The best value (train_inacc/train_loss) to check while determining 
            the best model which will be used to evaluate its performance on 
            the test set. Default is 'train_loss'.

        Returns
        -------
        dict
            Return train and test results dict.
            {
                'train' : {'preds': Tensor, 'target': Tensor, 'acc': Tensor},
                'test'  : ...
            }
        '''
        log_dir, writer, loger = self.reset_fitter(log_dir)

        # check the best model
        best_var = float('inf')
        best_net_param = deepcopy(self.net.state_dict())

        check_list = ['train_inacc', 'train_loss']
        if var_check not in check_list:
            s = ', '.join(check_list)
            raise ValueError(f'Parameter `var_check` only supports: {s} '
                             f'when training without val, but got {var_check}')

        # initialize dataloader
        train_loader = self._data_loader(*train_data, batch_size=self.batch_size)
        test_loader = self._data_loader(*test_data, batch_size=self.batch_size)
        ncls = len(cls_name)

        # start the training
        self.timer.start()
        loger.info(f'[Training...] - [{self.timer.ctime()}]')
        stop_criteria = ComposeStopCriteria({'Or': {
            'cri1' : {'MaxEpoch': {
                'max_epochs': max_epochs, 'var_name': 'epoch'
            }},
            'cri2': {'NoDecrease': {
                'numEpochs': no_increase_epochs, 'var_name': var_check
            }}
        }})
        self.exp_details['fit'] = {
            'type': 'fit_without_val',
            'var_check': var_check,
            'stop_criteria': str(stop_criteria)
        }
        monitors = {'epoch': 0, 'train_loss': float('inf'), 'train_inacc': 1}

        while not stop_criteria(monitors):

            # train one epoch
            self._run_one_epoch(train_loader)

            # evaluate the training and validation accuracy
            train_preds, train_target, train_loss = self._predict(train_loader)
            train_acc = self.get_acc(train_preds, train_target, ncls)
            monitors['train_inacc'] = 1 - train_acc
            monitors['train_loss'] = train_loss

            # store loss and acc
            writer.add_scalars('train', {'loss': train_loss, 'acc': train_acc},
                               monitors['epoch'])
            loger.info(f'-->Epoch : {monitors["epoch"] + 1}')
            loger.info(f'  \u21b3train Loss/Acc = {train_loss:.4f}/{train_acc:.4f}')

            # select best model
            if monitors[var_check] <= best_var:
                best_var = monitors[var_check]
                best_net_param = deepcopy(self.net.state_dict())

            monitors['epoch'] += 1

        writer.close()

        # report the checkpoint time of end and compute cost time
        loger.info(f'[Train Finish] - [{self.timer.ctime()}]')
        h, m, s = self.timer.stop()
        loger.info(f'Cost time = {h}H:{m}M:{s:.2f}S')

        # load the best model and evaulate this model in test_data
        self.net.load_state_dict(best_net_param)

        results = {}
        train_preds, train_target, train_loss = self._predict(train_loader)
        train_acc = self.get_acc(train_preds, train_target, ncls)
        results['train'] = {
            'preds': train_preds, 'target': train_target, 'acc': train_acc
        }
        test_preds, test_target, test_loss = self._predict(test_loader)
        test_acc = self.get_acc(test_preds, test_target, ncls)
        results['test'] = {
            'preds': test_preds, 'target': test_target, 'acc': test_acc
        }

        loger.info(f'Loss: train={train_loss:.4f} | test={test_loss:.4f}')
        loger.info(f'Acc:  train={train_acc:.4f} | test={test_acc:.4f}')

        # save the experiment details
        self.exp_details['results'] = results
        self.exp_details['best_net_param'] = best_net_param

        # store the training details
        with open(os.path.join(log_dir, f'train.pkl'), 'wb') as f:
            pickle.dump(self.exp_details, f)

        # store the best net model parameters
        model_path = os.path.join(log_dir, f'train_checkpoint_best.pth')
        torch.save(best_net_param, model_path)

        return results

    def fit_with_val(
        self,
        train_data : Union[tuple, list],
        val_data : Union[tuple, list],
        test_data : Union[tuple, list],
        log_dir : str,
        cls_name : Union[tuple, list],
        max_epochs_s1 : int = 1500,
        max_epochs_s2 : int = 600,
        no_increase_epochs : int = 200,
        second_stage : bool = True,
        var_check : str = 'val_inacc',
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
        train_data : tuple, list
            Dataset used for training. If type is tuple or list, dataset should
            be (data, labels).
        val_data : tuple, list
            Dataset used for validation. If type is tuple or list, dataset 
            should be (data, labels).
        test_data : tuple, list
            Dataset used to evaluate the model. If type is tuple or list, 
            dataset should be (data, labels).
        log_dir : str
            Directory location (support hierarchical folder structure) to save
            training log.
        cls_name : tuple, list
            The name of dataset labels.
        max_epochs_s1, max_epochs_s2 : int
            Maximum number of epochs in the x stage of training. Default is
            1500 and 600 respectively.
        no_increase_epochs : int
            Maximum number of consecutive epochs when the accuracy or loss of 
            the first-stage validation set has no relative improvement. Default
            is 200.
        second_stage : bool
            If True, two-stage training will be performed. Default is True.
        var_check : str
            The best value (val_inacc/val_loss) to check while determining the 
            best model which will be used for parameter initialization in the
            second stage of model training. Default is 'val_inacc'.

        Returns
        -------
        dict
            Return train, validation and test results dict.
            {
                'train' : {'preds': Tensor, 'target': Tensor, 'acc': Tensor},
                'test'  : ...,
                'val'   : ...
            }
        '''
        log_dir, writer, loger = self.reset_fitter(log_dir)

        # check the best model
        best_var = float('inf')
        best_net_param = deepcopy(self.net.state_dict())
        best_optim_param = deepcopy(self.optimizer.state_dict())

        check_list = ['val_inacc', 'val_loss']
        if var_check not in check_list:
            s = ', '.join(check_list)
            raise ValueError(f'Parameter `var_check` only supports: {s}'
                             f' when training with val, but got {var_check}')

        # initialize dataloader
        train_loader = self._data_loader(*train_data, batch_size=self.batch_size)
        val_loader = self._data_loader(*val_data, batch_size=self.batch_size)
        test_loader = self._data_loader(*test_data, batch_size=self.batch_size)
        ncls = len(cls_name)

        # start the training
        self.timer.start()
        loger.info(f'[Training...] - [{self.timer.ctime()}]')

        stop_criteria = ComposeStopCriteria({'Or': {
            'cri1': {'MaxEpoch': {
                'max_epochs': max_epochs_s1, 'var_name': 'epoch'
            }},
            'cri2': {'NoDecrease': {
                'numEpochs': no_increase_epochs, 'var_name': var_check
            }}
        }})
        self.exp_details['fit'] = {
            'type': 'fit_with_val',
            'var_check': var_check,
            'stopMon_1': str(stop_criteria)
        }
        monitors = {'epoch': 0, 'val_loss': float('inf'), 'val_inacc': 1,
                    'train_loss': float('inf'), 'global_epoch': 0}
        early_stop_reached, do_stop = False, False

        while not do_stop: 

            # train one epoch
            self._run_one_epoch(train_loader)

            # evaluate the training and validation accuracy
            train_preds, train_target, train_loss = self._predict(train_loader)
            train_acc = self.get_acc(train_preds, train_target, ncls)
            val_preds, val_target, val_loss = self._predict(val_loader)
            val_acc = self.get_acc(val_preds, val_target, ncls)
            monitors['val_inacc'] = 1 - val_acc
            monitors['val_loss'] = val_loss

            # store loss and acc
            writer.add_scalars('train', {'loss': train_loss, 'acc': train_acc}, 
                                monitors['global_epoch'])
            writer.add_scalars('val', {'loss': val_loss, 'acc': val_acc}, 
                                monitors['global_epoch'])
            # print the epoch info
            loger.info(f'-->Epoch : {monitors["epoch"] + 1}')
            loger.info(f'  \u21b3train Loss/Acc = {train_loss:.4f}/{train_acc:.4f}'
                        f' | val Loss/Acc = {val_loss:.4f}/{val_acc:.4f}')

            # select best model on Stage 1
            if monitors[var_check] <= best_var:
                best_var = monitors[var_check]
                best_net_param = deepcopy(self.net.state_dict())
                best_optim_param = deepcopy(self.optimizer.state_dict())
                monitors['train_loss'] = train_loss

            monitors['epoch'] += 1
            monitors['global_epoch'] += 1

            # check if to stop training
            if stop_criteria(monitors):
                # check whether to enter the second stage of training
                if second_stage and not early_stop_reached:
                    early_stop_reached = True
                    loger.info('[Early Stopping Reached] -> Continue '
                                    'training on full set.')
                    # load the best state
                    self.net.load_state_dict(best_net_param)
                    self.optimizer.load_state_dict(best_optim_param)

                    # Combine the train and val dataset
                    train_loader = self._data_loader(*train_data, *val_data,
                                                    batch_size=self.batch_size)

                    # update stop monitor and epoch
                    stop_criteria = ComposeStopCriteria({'Or': {
                        'cri1': {'MaxEpoch': {
                            'max_epochs': max_epochs_s2, 'var_name': 'epoch'
                        }},
                        'cri2': {'Smaller': {
                            'var': monitors['train_loss'], 'var_name': 'val_loss'
                        }}
                    }})
                    self.exp_details['fit']['stopMon_2'] = str(stop_criteria)
                    monitors['epoch'] = 0
                elif second_stage and early_stop_reached:
                    best_net_param = deepcopy(self.net.state_dict())
                    do_stop = True
                else:
                    do_stop = True

        writer.close()

        # report the checkpoint time of end and compute cost time
        loger.info(f'[Train Finish] - [{self.timer.ctime()}]')
        h, m, s = self.timer.stop()
        loger.info(f'Cost Time = {h}H:{m}M:{s:.2f}S')

        # load the best model and evaulate this model in test_data
        self.net.load_state_dict(best_net_param)

        results = {}
        train_preds, train_target, train_loss = self._predict(train_loader)
        train_acc = self.get_acc(train_preds, train_target, ncls)
        results['train'] = {
            'preds': train_preds, 'target': train_target, 'acc': train_acc
        }
        val_preds, val_target, val_loss = self._predict(val_loader)
        val_acc = self.get_acc(val_preds, val_target, ncls)
        results['val'] = {
            'preds': val_preds, 'target': val_target, 'acc': val_acc
        }
        test_preds, test_target, test_loss = self._predict(test_loader)
        test_acc = self.get_acc(test_preds, test_target, ncls)
        results['test'] = {
            'preds': test_preds, 'target': test_target, 'acc': test_acc
        }

        loger.info(f'Loss: train={train_loss:.4f} | val={val_loss:.4f} | '
                   f'test={test_loss:.4f}')
        loger.info(f'Acc:  train={train_acc:.4f} | val={val_acc:.4f} | '
                   f'test={test_acc:.4f}')

        # save the experiment details
        self.exp_details['results'] = results
        self.exp_details['best_net_param'] = best_net_param

        # store the training details
        with open(os.path.join(log_dir, f'train.pkl'), 'wb') as f:
            pickle.dump(self.exp_details, f)

        # store the best net model parameters
        model_path = os.path.join(log_dir, f'train_checkpoint_best.pth')
        torch.save(best_net_param, model_path)

        return results

    def __repr__(self) -> str:
        '''Trainer details.
        '''
        s = '[Network architecture]:\n' + self.net_arch + '\n'
        s += f'[Loss function]: {self.loss_fn_type}\n'
        if self.loss_fn_args:
            s += f'[loss_fn Args]: {self.lr_sch_args}\n'
        s += f'[Optimizer]: {self.optimizer_type}\n'
        s += f'[Learning rate]: {self.lr}\n'
        if self.optimizer_args:
            s += f'[Optim Args]: {self.optimizer_args}\n'
        if self.lr_sch_type:
            s += f'[Lr scheduler]: {self.lr_sch_type}\n'
            if self.lr_sch_args:
                s += f'[lr_sch Args]: {self.lr_sch_args}\n'
        s += f'[Grad Acc]: {self.grad_acc}\n'
        s += f'[Batch Size]: {self.batch_size}\n'
        return s
