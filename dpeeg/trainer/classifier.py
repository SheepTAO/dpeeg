# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

import os
from abc import abstractmethod, ABC
from copy import deepcopy

import torch
import torch.nn as nn
from torch import optim
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.backends import cudnn
from torchinfo import summary
from typing import Literal, Type
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchmetrics.functional.classification.accuracy import multiclass_accuracy
from torchmetrics.aggregation import MeanMetric, CatMetric

from .base import Trainer
from ..tools import Logger, Timer
from ..utils import DPEEG_SEED
from ..transforms.functions import to_tensor
from .stopcriteria import ComposeStopCriteria
from .utils import get_device, get_device_name, model_depth
from ..utils import _set_torch_seed, mapping_to_str
from ..datasets.base import EEGData


cudnn.benchmark = False
cudnn.deterministic = True


class BaseClassifier(Trainer, ABC):
    """Classification model trainer.

    Generate a trainer to test the performance of the same network on
    different datasets.
    """

    def __init__(
        self,
        model: Module,
        loss_fn: str | type[Module] = "NLLLoss",
        loss_fn_args: dict | None = None,
        optimizer: str | type[Optimizer] = "Adam",
        optimizer_args: dict | None = None,
        lr: float = 1e-3,
        lr_sch: str | type[LRScheduler] | None = None,
        lr_sch_args: dict | None = None,
        grad_acc: int = 1,
        batch_size: int = 32,
        nGPU: int = 0,
        seed: int = DPEEG_SEED,
        keep_data_gpu: bool = True,
        depth: int | None = None,
        data_size: tuple | list | None = None,
        verbose: int | str = "INFO",
    ) -> None:
        super().__init__(model)
        self.model = model

        self.loger = Logger("dpeeg_train", clevel=verbose)
        self.timer = Timer()

        # init trainer
        self.device = get_device(nGPU)
        self.loger.info(
            f"Model will be trained on {self.device} "
            f"({get_device_name(self.device)})"
        )
        self.model.to(self.device)

        _set_torch_seed(seed)
        self.loger.info(f"Set torch random seed = {seed}")

        # summarize model structure
        self.model_arch = str(model) + "\n"
        depth = model_depth(self.model) if depth is None else depth
        self.model_arch += str(summary(model, data_size, depth=depth))
        self.loger.info(self.model_arch)

        # the type of optimizer, etc. selected
        self.loss_fn_type = loss_fn
        self.optimizer_type = optimizer
        self.lr_sch_type = lr_sch

        # save additional parameters
        self.loss_fn_args = loss_fn_args if loss_fn_args else {}
        self.optimizer_args = optimizer_args if optimizer_args else {}
        self.lr_sch_args = lr_sch_args if lr_sch_args else {}
        # --- others
        self.optimizer_args["lr"] = lr
        self.grad_acc = grad_acc
        self.batch_size = batch_size
        self.seed = seed
        self.keep_data_gpu = keep_data_gpu
        self.verbose = verbose

        # set experimental details
        self.train_details = {
            "type": self.__class__.__name__,
            "train_param": {
                "seed": seed,
                "loss_fn": str(loss_fn),
                "loss_fn_args": loss_fn_args,
                "optimizer": str(optimizer),
                "optimizer_args": optimizer_args,
                "lr_sch": str(lr_sch),
                "lr_sch_args": lr_sch_args,
                "batch_size": batch_size,
                "grad_acc": grad_acc,
            },
            "orig_model_param": deepcopy(self.model.state_dict()),
        }

    def fit_epoch(self, train_loader: DataLoader) -> None:
        """Fit one epoch to train model.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader used for training.
        """
        # set the model in training mode
        self.model.train()

        # iterater over all the data
        with torch.enable_grad():
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.to(self.device), label.to(self.device)
                out = self.model(data)
                loss = self.loss_fn(out, label)
                loss.backward()
                # gradient accumulation
                if (idx + 1) % self.grad_acc == 0:
                    # 1 - update parameters
                    self.optimizer.step()
                    # 2 - zero the parameter gradients
                    self.optimizer.zero_grad()
            # update lr
            # Note: Learning rate scheduling should be applied after optimizer’s update
            if self.lr_sch:
                self.lr_sch.step()

    def predict(
        self,
        data_loader: DataLoader,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict the class of the input data.

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
        """
        # set the model in the eval mode
        self.model.eval()

        loss_sum = MeanMetric()
        preds, target = CatMetric(), CatMetric()

        # iterate over all the data
        with torch.no_grad():
            for data, label in data_loader:
                data, label = data.to(self.device), label.to(self.device)
                out = self.model(data)
                loss = self.loss_fn(out, label)
                loss_sum.update(loss.item(), data.size(0))
                # convert the output of soft-max to class label
                # save preds and actual label
                out = out[0] if isinstance(out, tuple) else out
                preds.update(torch.argmax(out, dim=1).detach().cpu())
                target.update(label.cpu())
        return preds.compute(), target.compute(), loss_sum.compute()

    def data_loader(self, *datasets: EEGData) -> DataLoader:
        """Wrap multiple sets of data and labels and return DataLoader.

        Parameters
        ----------
        datasets : sequence of EEGData
            Sequence of EEGData. Multiple EEGData will be concatenated.
        """
        if len(datasets) == 0:
            raise ValueError("At least one dataset required as input.")

        # dataset wrapping tensors
        td = []
        for dataset in datasets:
            data, label = to_tensor(dataset["edata"], dataset["label"])
            if self.keep_data_gpu:
                data, label = data.to(self.device), label.to(self.device)
            td.append(TensorDataset(data, label))
        td = ConcatDataset(td)

        return DataLoader(td, self.batch_size, True)

    def _reset_fitter(self, log_dir: str) -> tuple[str, SummaryWriter, Logger]:
        """Reset the relevant parameters of the fitter.

        Reset the model's training parameters, learning rate schedule and
        optimizer etc. to their initialized state.

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
        """
        # reset parameters of nn.Moudle
        self.model.load_state_dict(self.train_details["orig_model_param"])

        # create loss function
        if isinstance(self.loss_fn_type, str):
            self.loss_fn = getattr(nn, self.loss_fn_type)(**self.loss_fn_args)
        else:
            self.loss_fn = self.loss_fn_type(**self.loss_fn_args)

        # create optimizer
        if isinstance(self.optimizer_type, str):
            self.optimizer = getattr(optim, self.optimizer_type)(
                self.model.parameters(), **self.optimizer_args
            )
        elif isinstance(self.optimizer_type, Optimizer):
            self.optimizer = self.optimizer_type(
                self.model.parameters(), self.optimizer_args
            )
        else:
            raise TypeError(
                f"Optimizer type ({type(self.optimizer_type)}) is not supported."
            )

        # create lr_scheduler
        if self.lr_sch_type is None:
            self.lr_sch = None
        elif isinstance(self.lr_sch_type, str):
            self.lr_sch = getattr(optim.lr_scheduler, self.lr_sch_type)(
                self.optimizer, **self.lr_sch_args
            )
        elif isinstance(self.lr_sch_type, LRScheduler):
            self.lr_sch = self.lr_sch_type(self.optimizer, **self.lr_sch_args)
        else:
            raise TypeError(f"lr_sch type ({type(self.lr_sch_type)}) is not supported.")

        # create log writer
        log_dir = os.path.abspath(log_dir)
        writer = SummaryWriter(log_dir)
        loger = Logger(
            log_dir,
            path=os.path.join(log_dir, "running.log"),
            flevel="INFO",
            clevel=self.verbose,
        )

        return log_dir, writer, loger

    def get_acc(self, preds: Tensor, target: Tensor, ncls: int) -> Tensor:
        """Easy for program to caculate the accuarcy."""
        return multiclass_accuracy(preds, target, ncls, "micro")

    def __repr__(self) -> str:
        s = "[Model architecture]:\n" + self.model_arch + "\n"

        s += f"[Loss function]: {self.loss_fn_type}"
        if self.loss_fn_args:
            s += f"({mapping_to_str(self.loss_fn_args)})\n"
        else:
            s += "\n"

        s += f"[Optimizer]: {self.optimizer_type}"
        if self.optimizer_args:
            s += f"({mapping_to_str(self.optimizer_args)})\n"
        else:
            s += "\n"

        if self.lr_sch_type:
            s += f"[Lr scheduler]: {self.lr_sch_type}"
            if self.lr_sch_args:
                s += f"({mapping_to_str(self.lr_sch_args)})\n"
            else:
                s += "\n"

        s += f"[Grad Acc]: {self.grad_acc}\n"
        s += f"[Batch Size]: {self.batch_size}\n"
        s += f"[Seed]: {self.seed}\n"

        return s


class Classifier(BaseClassifier):
    """Classifier training.

    During different training model processes, early stopping mechanisms can be
    executed using the training set (validation set not required) to select the
    model.

    Parameters
    ----------
    model : Module
        Inherit Module and should define the forward method. The first
        parameter returned by model forward propagation is the prediction.
    loss_fn : str, Type[Module]
        Name of the loss function from `torch.nn` which will be used for
        training. If Module, means using a custom loss function.
        Note: custom loss_fn is a class (not an instance), and its
        initialization list is `(**loss_fn_args)`.
    loss_fn_args : dict, optional
        Additional arguments to be passed to the loss function.
    optimizer : str, Type[Optimizer]
        Name of the optimization function from torch.optim which will be used
        for training. If Optimizer, means using a custom optimizer.
        Note: custom optimizer is a class (not an instance), and its
        initialization list is `(model, lr=lr, **optimizer_args)`.
    optimizer_args : dict, optional
        Additional arguments to be passed to the optimization function.
    lr : float
        Learning rate.
    lr_sch : str, Type[LRScheduler], optional
        Name of the learning scheduler from `torch.optim.lr_scheduler` which
        will be used for training. If LRScheduler, means using a custom
        learning scheduler.
        Note: custom learning scheduler is a class (not an instance), and its
        initialization list is `(optimizer, **lr_sch_args)`.
    lr_sch_args : dict, optional
        Additional arguments to be passed to the lr_scheduler function.
    grad_acc : int
        Aradient accumulation.
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum number of epochs in training.
    no_increase_epochs : int
        Maximum number of consecutive epochs when the accuracy or loss of the
        training set has no relative improvement.
    var_check : str
        The best value (train_inacc/train_loss) to check while determining the
        best model which will be used to evaluate its performance on the test.
    load_best_state : bool
        If `True`, the best model parameters will be used for evaluation.
    nGPU : int
        Select the gpu id to train. If the GPU is not available then the CPU is
        used.
    seed : int
        Select random seed for review.
    keep_data_gpu : bool
        Keep the dataset on the GPU to avoid the time consumption of data
        migration. Please adjust according to the personal GPU memory.
    data_size : tuple, list, optional
        Output the structure of the network model according to the input
        dimension if the `data_size` is given.
    depth : int, optional
        Depth of nested layers to display.
    verbose : int, str
        The log level of console. Default is INFO. Mainly used for debug.
    """

    def __init__(
        self,
        model: Module,
        loss_fn: str | Type[Module] = "NLLLoss",
        loss_fn_args: dict | None = None,
        optimizer: str | Type[Optimizer] = "Adam",
        optimizer_args: dict | None = None,
        lr: float = 0.001,
        lr_sch: str | Type[LRScheduler] | None = None,
        lr_sch_args: dict | None = None,
        grad_acc: int = 1,
        batch_size: int = 32,
        max_epochs: int = 1000,
        no_increase_epochs: int = 200,
        var_check: Literal["train_loss", "train_inacc"] = "train_loss",
        load_best_state: bool = True,
        nGPU: int = 0,
        seed: int = DPEEG_SEED,
        keep_data_gpu: bool = True,
        depth: int | None = None,
        data_size: tuple | list | None = None,
        verbose: int | str = "INFO",
    ) -> None:
        super().__init__(
            model,
            loss_fn,
            loss_fn_args,
            optimizer,
            optimizer_args,
            lr,
            lr_sch,
            lr_sch_args,
            grad_acc,
            batch_size,
            nGPU,
            seed,
            keep_data_gpu,
            depth,
            data_size,
            verbose,
        )
        self.max_epochs = max_epochs
        self.no_increase_epochs = no_increase_epochs
        self.var_check = var_check
        self.load_best_state = load_best_state
        self.train_details.update(
            {
                "train_param": {
                    "max_epochs": max_epochs,
                    "no_increase_epochs": no_increase_epochs,
                    "var_check": var_check,
                    "load_best_state": load_best_state,
                }
            }
        )

    def fit(
        self,
        trainset: EEGData,
        testset: EEGData,
        log_dir: str,
    ) -> dict[str, dict[str, Tensor]]:
        """Fit the model.

        Parameters
        ----------
        trainset : EEGData
            Dataset used for training.
        testset : EEGData
            Dataset used to evaluate the model.
        log_dir : str
            The path to save the training log.

        Returns
        -------
        result : dict
            Returns the training set and test set results (including true
            labels, predicted labels and accuracy).
        """
        log_dir, writer, loger = self._reset_fitter(log_dir)

        # check the best model
        best_var = float("inf")
        best_model_param = deepcopy(self.model.state_dict())

        # initialize dataloader
        train_loader = self.data_loader(trainset)
        test_loader = self.data_loader(testset)
        train_ncls = trainset.ncls
        test_ncls = testset.ncls

        # start the training
        self.timer.start()
        loger.info(f"[Training...] - [{self.timer.ctime()}]")
        loger.info(f"[Train/Test] - [{trainset.trials()}/{testset.trials()}]")

        stopcri = ComposeStopCriteria(
            {
                "Or": {
                    "cri1": {
                        "MaxEpoch": {"max_epochs": self.max_epochs, "var_name": "epoch"}
                    },
                    "cri2": {
                        "NoDecrease": {
                            "num_epochs": self.no_increase_epochs,
                            "var_name": self.var_check,
                        }
                    },
                }
            }
        )
        monitors = {"epoch": 0, "train_loss": float("inf"), "train_inacc": 1}

        while not stopcri(monitors):

            # train one epoch
            self.fit_epoch(train_loader)
            monitors["epoch"] += 1

            # evaluate the training and validation accuracy
            train_preds, train_target, train_loss = self.predict(train_loader)
            train_acc = self.get_acc(train_preds, train_target, train_ncls)
            monitors["train_inacc"] = 1 - train_acc
            monitors["train_loss"] = train_loss

            # store loss and acc
            writer.add_scalars(
                "train", {"loss": train_loss, "acc": train_acc}, monitors["epoch"]
            )
            loger.info(f'-->Epoch : {monitors["epoch"]}')
            loger.info(f"  \u21b3train Loss/Acc = {train_loss:.4f}/{train_acc:.4f}")

            # select best model
            if self.load_best_state and monitors[self.var_check] <= best_var:
                best_var = monitors[self.var_check]
                best_model_param = deepcopy(self.model.state_dict())

        writer.close()
        if not self.load_best_state:
            best_model_param = deepcopy(self.model.state_dict())

        # report the checkpoint time of end and compute cost time
        h, m, s = self.timer.stop()
        loger.info(f"[Train Finish] - [Cost Time = {h}H:{m}M:{s:.2f}S]")

        # load the best model and evaulate this model in testset
        self.model.load_state_dict(best_model_param)

        results = {}
        train_preds, train_target, train_loss = self.predict(train_loader)
        train_acc = self.get_acc(train_preds, train_target, train_ncls)
        results["train"] = {
            "preds": train_preds,
            "target": train_target,
            "acc": train_acc,
        }
        test_preds, test_target, test_loss = self.predict(test_loader)
        test_acc = self.get_acc(test_preds, test_target, train_ncls)
        results["test"] = {"preds": test_preds, "target": test_target, "acc": test_acc}

        loger.info(f"Loss: Train={train_loss:.4f} | Test={test_loss:.4f}")
        loger.info(f"Acc:  Train={train_acc:.4f} | Test={test_acc:.4f}")

        self.train_details["results"] = results
        self.train_details["best_model_param"] = best_model_param

        # store the training details
        train_details_path = os.path.join(log_dir, f"train_details.pt")
        torch.save(self.train_details, train_details_path)

        # store the best model parameters
        best_checkpoiont_path = os.path.join(log_dir, f"best_checkpoint.pth")
        torch.save(best_model_param, best_checkpoiont_path)

        return results


class ClassifierTwoStage(BaseClassifier):
    """Two-stage classifier training.

    Two-stage training strategy was used. In the first stage, the model was
    trained using only the training set with the early stopping criteria
    whereby the validation set accuracy and loss was monitored and training was
    stopped if there was no increase in the validation set accuracy (or loss)
    for consecutive 200 epochs. After reaching the stopping criteria, network
    parameters with the best validation set accuracy (or loss) were restored.
    In the second stage, the model was trained with the complete training data
    (train + validation set). The second stage training was stopped when the
    validation set loss reduced below the first stage training set loss.

    Parameters
    ----------
    model : Module
        Inherit Module and should define the forward method. The first
        parameter returned by model forward propagation is the prediction.
    loss_fn : str, Type[Module]
        Name of the loss function from `torch.nn` which will be used for
        training. If Module, means using a custom loss function.
        Note: custom loss_fn is a class (not an instance), and its
        initialization list is `(**loss_fn_args)`.
    loss_fn_args : dict, optional
        Additional arguments to be passed to the loss function.
    optimizer : str, Type[Optimizer]
        Name of the optimization function from torch.optim which will be used
        for training. If Optimizer, means using a custom optimizer.
        Note: custom optimizer is a class (not an instance), and its
        initialization list is `(model, lr=lr, **optimizer_args)`.
    optimizer_args : dict, optional
        Additional arguments to be passed to the optimization function.
    lr : float
        Learning rate.
    lr_sch : str, Type[LRScheduler], optional
        Name of the learning scheduler from `torch.optim.lr_scheduler` which
        will be used for training. If LRScheduler, means using a custom
        learning scheduler.
        Note: custom learning scheduler is a class (not an instance), and its
        initialization list is `(optimizer, **lr_sch_args)`.
    lr_sch_args : dict, optional
        Additional arguments to be passed to the lr_scheduler function.
    grad_acc : int
        Aradient accumulation.
    batch_size : int
        Mini-batch size.
    max_epochs_s1, max_epochs_s2 : int
        Maximum number of epochs in the x stage of training.
    no_increase_epochs : int
        Maximum number of consecutive epochs when the accuracy or loss of the
        first-stage validation set has no relative improvement.
    second_stage : bool
        If `True`, two-stage training will be performed.
    load_best_state : bool
        If `True`, two-stage will retrain based on the best state in first-
        stage. Otherwise, two-stage will retain based on the last state of the
        first-stage.
    var_check : str
        The best value (valid_inacc/valid_loss) to check while determining the
        best state which will be used for parameter initialization in the
        second stage of model training.
    cls_name : list of str
        The name of dataset labels.
    nGPU : int
        Select the gpu id to train. If the GPU is not available then the CPU is
        used.
    seed : int
        Select random seed for review.
    keep_data_gpu : bool
        Keep the dataset on the GPU to avoid the time consumption of data
        migration. Please adjust according to the personal GPU memory.
    data_size : tuple, list, optional
        Output the structure of the network model according to the input
        dimension if the `data_size` is given.
    depth : int, optional
        Depth of nested layers to display.
    verbose : int, str
        The log level of console. Default is INFO. Mainly used for debug.
    """

    def __init__(
        self,
        model: Module,
        loss_fn: str | Type[Module] = "NLLLoss",
        loss_fn_args: dict | None = None,
        optimizer: str | Type[Optimizer] = "Adam",
        optimizer_args: dict | None = None,
        lr: float = 0.001,
        lr_sch: str | Type[LRScheduler] | None = None,
        lr_sch_args: dict | None = None,
        grad_acc: int = 1,
        batch_size: int = 32,
        max_epochs_s1: int = 1500,
        max_epochs_s2: int = 600,
        no_increase_epochs: int = 200,
        second_stage: bool = True,
        var_check: Literal["valid_inacc", "valid_loss"] = "valid_inacc",
        load_best_state: bool = True,
        nGPU: int = 0,
        seed: int = DPEEG_SEED,
        keep_data_gpu: bool = True,
        depth: int | None = None,
        data_size: tuple | list | None = None,
        verbose: int | str = "INFO",
    ) -> None:
        super().__init__(
            model,
            loss_fn,
            loss_fn_args,
            optimizer,
            optimizer_args,
            lr,
            lr_sch,
            lr_sch_args,
            grad_acc,
            batch_size,
            nGPU,
            seed,
            keep_data_gpu,
            depth,
            data_size,
            verbose,
        )
        self.max_epochs_s1 = max_epochs_s1
        self.max_epochs_s2 = max_epochs_s2
        self.no_increase_epochs = no_increase_epochs
        self.second_stage = second_stage
        self.var_check = var_check
        self.load_best_state = load_best_state
        self.train_details.update(
            {
                "train_param": {
                    "max_epochs_s1": max_epochs_s1,
                    "max_epochs_s2": max_epochs_s2,
                    "no_increase_epochs": no_increase_epochs,
                    "second_stage": second_stage,
                    "var_check": var_check,
                    "load_best_state": load_best_state,
                }
            }
        )

    def fit(
        self,
        trainset: EEGData,
        validset: EEGData,
        testset: EEGData,
        log_dir: str,
    ):
        """Fit the model.

        Parameters
        ----------
        trainset : EEGData
            Dataset used for training.
        validset : EEGData
            Dataset used for validation.
        testset : EEGData
            Dataset used to evaluate the model.
        log_dir : str
            The path to save the training log.

        Returns
        -------
        dict
            Returns the training set, validation set, and test set results
            (including true labels, predicted labels and accuracy).
        """
        log_dir, writer, loger = self._reset_fitter(log_dir)

        # check the best model
        best_var = float("inf")
        best_model_param = deepcopy(self.model.state_dict())
        best_optim_param = deepcopy(self.optimizer.state_dict())

        # initialize dataloader
        train_loader = self.data_loader(trainset)
        valid_loader = self.data_loader(validset)
        test_loader = self.data_loader(testset)
        train_ncls = trainset.ncls
        valid_ncls = validset.ncls
        test_ncls = testset.ncls

        # start the training
        self.timer.start()
        loger.info(f"[Training...] - [{self.timer.ctime()}]")
        loger.info(
            f"[Train/Valid/Test] - "
            f"[{trainset.trials()}/{validset.trials()}/{testset.trials()}]"
        )

        stopcri = ComposeStopCriteria(
            {
                "Or": {
                    "cri1": {
                        "MaxEpoch": {
                            "max_epochs": self.max_epochs_s1,
                            "var_name": "epoch",
                        }
                    },
                    "cri2": {
                        "NoDecrease": {
                            "num_epochs": self.no_increase_epochs,
                            "var_name": self.var_check,
                        }
                    },
                }
            }
        )
        self.train_details["fit"] = {
            "type": "fit_with_val",
            "var_check": self.var_check,
            "stopcri_1": str(stopcri),
        }
        monitors = {
            "epoch": 0,
            "valid_loss": float("inf"),
            "valid_inacc": 1,
            "global_epoch": 0,
            "best_epoch": -1,
            "best_train_loss": float("inf"),
        }
        load_best_state = self.load_best_state
        early_stop_reached, do_stop = False, False

        while not do_stop:

            # train one epoch
            self.fit_epoch(train_loader)
            monitors["epoch"] += 1
            monitors["global_epoch"] += 1

            # evaluate the training and validation accuracy
            train_preds, train_target, train_loss = self.predict(train_loader)
            train_acc = self.get_acc(train_preds, train_target, train_ncls)
            valid_preds, valid_target, valid_loss = self.predict(valid_loader)
            valid_acc = self.get_acc(valid_preds, valid_target, valid_ncls)
            monitors["valid_inacc"] = 1 - valid_acc
            monitors["valid_loss"] = valid_loss

            # store loss and acc
            writer.add_scalars(
                "train",
                {"loss": train_loss, "acc": train_acc},
                monitors["global_epoch"],
            )
            writer.add_scalars(
                "valid",
                {"loss": valid_loss, "acc": valid_acc},
                monitors["global_epoch"],
            )
            # print the epoch info
            loger.info(f'-->Epoch : {monitors["epoch"]}')
            loger.info(
                f"  \u21b3train Loss/Acc = {train_loss:.4f}/{train_acc:.4f}"
                f" | valid Loss/Acc = {valid_loss:.4f}/{valid_acc:.4f}"
            )

            # select best model on Stage 1
            if load_best_state and monitors[self.var_check] <= best_var:
                best_var = monitors[self.var_check]
                best_model_param = deepcopy(self.model.state_dict())
                best_optim_param = deepcopy(self.optimizer.state_dict())
                monitors["best_train_loss"] = train_loss
                monitors["best_epoch"] = monitors["epoch"]

            # check whether to stop training
            if stopcri(monitors):
                # check whether to enter the second stage of training
                if self.second_stage and not early_stop_reached:
                    early_stop_reached = True
                    epoch = monitors["epoch"]

                    # load the best state
                    if load_best_state:
                        self.model.load_state_dict(best_model_param)
                        self.optimizer.load_state_dict(best_optim_param)
                        train_loss = monitors["best_train_loss"]
                        epoch = monitors["best_epoch"]

                    loger.info("[Early Stopping Reached] -> Training on full set.")
                    loger.info(f"[Epoch = {epoch} | Loss = {train_loss:.4f}]")

                    # combine the train and valid dataset
                    train_loader = self.data_loader(trainset, validset)

                    # update stop monitor and epoch
                    stopcri = ComposeStopCriteria(
                        {
                            "Or": {
                                "cri1": {
                                    "MaxEpoch": {
                                        "max_epochs": self.max_epochs_s2,
                                        "var_name": "epoch",
                                    }
                                },
                                "cri2": {
                                    "Smaller": {
                                        "var": train_loss,
                                        "var_name": "valid_loss",
                                    }
                                },
                            }
                        }
                    )
                    self.train_details["fit"]["stopcri_2"] = str(stopcri)
                    monitors["epoch"] = 0
                    load_best_state = False
                elif self.second_stage and early_stop_reached:
                    do_stop = True
                    best_model_param = deepcopy(self.model.state_dict())
                # no second stage
                else:
                    do_stop = True
                    if not load_best_state:
                        best_model_param = deepcopy(self.model.state_dict())

        writer.close()

        # report the checkpoint time of end and compute cost time
        h, m, s = self.timer.stop()
        loger.info(f"[Train Finish] - [Cost Time = {h}H:{m}M:{s:.2f}S]")

        # load the best model and evaulate this model in testset
        self.model.load_state_dict(best_model_param)

        results = {}
        train_preds, train_target, train_loss = self.predict(train_loader)
        train_acc = self.get_acc(train_preds, train_target, train_ncls)
        results["train"] = {
            "preds": train_preds,
            "target": train_target,
            "acc": train_acc,
        }
        valid_preds, valid_target, valid_loss = self.predict(valid_loader)
        valid_acc = self.get_acc(valid_preds, valid_target, valid_ncls)
        results["valid"] = {
            "preds": valid_preds,
            "target": valid_target,
            "acc": valid_acc,
        }
        test_preds, test_target, test_loss = self.predict(test_loader)
        test_acc = self.get_acc(test_preds, test_target, test_ncls)
        results["test"] = {
            "preds": test_preds,
            "target": test_target,
            "acc": test_acc,
        }

        loger.info(
            f"Loss: Train={train_loss:.4f} | Valid={valid_loss:.4f} | "
            f"test={test_loss:.4f}"
        )
        loger.info(
            f"Acc:  Train={train_acc:.4f} | Valid={valid_acc:.4f} | "
            f"Test={test_acc:.4f}"
        )

        self.train_details["results"] = results
        self.train_details["best_model_param"] = best_model_param

        # store the training details
        train_details_path = os.path.join(log_dir, f"train_details.pt")
        torch.save(self.train_details, train_details_path)

        # store the best model
        best_checkpoiont_path = os.path.join(log_dir, f"best_checkpoint.pth")
        torch.save(best_model_param, best_checkpoiont_path)

        return results
