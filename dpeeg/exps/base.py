# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from pathlib import Path
from abc import ABC, abstractmethod

import torch
from torchmetrics.aggregation import CatMetric, MeanMetric

import dpeeg
from ..datasets.base import SplitEEGData, BaseDataset, BaseData
from ..transforms.base import Transforms
from ..trainer.base import Trainer
from ..trainer.classifier import BaseClassifier
from ..tools import Logger, Timer, Filer
from ..utils import DPEEG_DIR, iterable_to_str, _format_log, _format_log_kv


class Experiment(ABC):
    """Experiment base class.

    Parameters
    ----------
    repr : str
        Basic information on the experiment method.
    trainer : Trainer
        Trainer used for training module on dataset.
    out_folder : str, optional
        Store all experimental results in a folder named with the model class
        name in the specified folder. Default is
        '~/dpeeg/out/model/exp/dataset/timestamp'.
    timestamp : bool
        Output folders are timestamped.
    verbose : int, str
        The log level of console. Default is INFO. Mainly used for debugging.

    Notes
    -----
    The training results of all models for each subject will be saved under the
    `out_folder` directory.
    """

    def __init__(
        self,
        repr: dict,
        trainer: Trainer,
        out_folder: str | None = None,
        timestamp: bool = True,
        verbose: int | str = "INFO",
    ) -> None:
        repr.pop("trainer")
        self._repr = repr
        self.trainer = trainer
        self.timestamp = timestamp
        self.verbose = verbose

        # create logger and timer
        self.logger = Logger("dpeeg_exp", clevel=verbose)
        self.timer = Timer()

        # set output folder
        net = trainer.model.__class__.__name__
        exp = self.__class__.__name__
        self.out_folder = (
            Path(out_folder).absolute().joinpath(net, exp)
            if out_folder
            else Path(DPEEG_DIR).joinpath("out", net, exp)
        )

    @abstractmethod
    def _run(self) -> dict:
        """Reconstruct different experimental processes according to different
        experimental designs.
        """
        pass

    def run(
        self,
        dataset: BaseDataset,
        dataset_name: str | None = None,
        transforms: Transforms | None = None,
        desc: str | None = None,
    ) -> dict:
        """Train models separately for each subject.

        This function will internally call the `_run_sub` function for each
        subject, and save the final results together.

        Parameters
        ----------
        dataset : EEG Data or Dataset
            The dataset used for the experimental test.
        dataset_name : str, optional
            The dataset name to use. If ``None``, The default name of the
            dataset will be used as the folder to save experimental results.
        transforms : Transforms, optional
            Apply pre-transforms on dataset. Transformations will be apply
            during the experiment on each subject's dataset. The rationable
            behind this method lies in deferring data manipulation, especially
            for certain transformations that could potentially enlarge the
            dataset's memory footprint. This delay allows for the execution of
            data manipulation after subject-independent experiment have
            concatenated the relevant data (Time for Space) or the experiment
            subject are ready, mitigating the risk of memory overflow.
        desc : str, optional
            Add a short description to the current experiment.

        Returns
        -------
        Return a dict of all subjects and corresponding experimental results.
        """

        if dataset_name:
            self.data_folder = self.out_folder / dataset_name
        else:
            self.data_folder = self.out_folder / dataset._repr["_obj_name"]

        if self.timestamp:
            self.data_folder = self.data_folder / Timer.cdate()
        self.data_folder.mkdir(parents=True, exist_ok=False)
        self.logger.info(f"Results saved in `{self.data_folder}`")

        self.dataset = dataset
        self.transforms = transforms

        self.filer = Filer(self.data_folder / "summary.txt")
        self.filer.write(f"[Start Time]: {self.timer.ctime()}\n")
        self.filer.write(f"[DPEEG Version]: {dpeeg.__version__}\n")
        self.filer.write(f"[Description]: {desc}\n")
        self.filer.write(str(dataset) + "\n")
        self.filer.write(_format_log_kv("Transforms", transforms) + "\n")
        self.filer.write(_format_log_kv("Trainer", self.trainer) + "\n")
        self.filer.write(str(self) + "\n")

        self.timer.start()
        self.logger.info("=" * 50)

        # Start the experiment for all subjects.
        results = self._run()

        h, m, s = self.timer.stop()
        torch.save(results, self.data_folder / f"results.pt")
        self.logger.info(
            f"\n[All Subjects Finished] - [Cost Time = {h}H:{m}M:{s:.2f}S]"
        )
        self.logger.info("=" * 50)

        return results

    def __repr__(self) -> str:
        return _format_log(self._repr)


class ClsExp(Experiment, ABC):
    """Base class for classification experiments."""

    @abstractmethod
    def _run_sub_classifier(self, *args, **kwargs) -> dict:
        pass

    @abstractmethod
    def _run_sub_classifier_two_stage(self, *args, **kwargs) -> dict:
        pass

    def __init__(
        self,
        repr: dict,
        trainer: BaseClassifier,
        out_folder: str | None = None,
        timestamp: bool = True,
        verbose: int | str = "INFO",
    ) -> None:
        super().__init__(
            repr=repr,
            trainer=trainer,
            out_folder=out_folder,
            timestamp=timestamp,
            verbose=verbose,
        )

        trainer_type = type(trainer).__name__
        trainer_list = {
            "Classifier": self._run_sub_classifier,
            "ClassifierTwoStage": self._run_sub_classifier_two_stage,
        }
        if trainer_type not in list(trainer_list.keys()):
            raise TypeError(
                f"Trainer type {trainer_type} is not supported, "
                f"only {iterable_to_str(trainer_list.keys())} are supported."
            )
        self._run_sub_func = trainer_list[trainer_type]

    def _trans_eegdata(self, eegdata: BaseData) -> SplitEEGData:
        """Apply pre-transforms on eegdata.

        Raises
        ------
        TypeError
            If the eegdata is not split after transformed.
        """
        if self.transforms is not None:
            eegdata = self.transforms(eegdata)

        if not isinstance(eegdata, SplitEEGData):
            raise TypeError("The eegdata is not split.")

        return eegdata

    def _process_sub_dataset(self, subject: int):
        """Preprocess each subject's dataset.

        Different preprocessing operations are performed on the dataset accord-
        ing to different experimental requirement. By default, eegdata for each
        subject in the dataset is returned.
        """
        return self.dataset[subject]

    @abstractmethod
    def _run_sub(self, eegdata: BaseData, sub_folder: Path):
        """Train a model on the specified subject data.

        This function will be called by `_run` function to conduct experiments
        on the data of each individual subject. Reconstruct the model training
        process according to different experimental requirements.

        Parameters
        ----------
        subject : int
            Subject of the experiment. Create a subdirectory of `out_folder` to
            store all yield results during subject training.
        eegdata : eegdata
            Subject eegdata. Adjust the subject eegdata according to different
            experiments.

        Returns
        -------
        result : dict
            Contains the `acc`, `preds`, `target` and detailed results.
        """
        pass

    def _run(self) -> dict:
        result = {}
        acc_metric = MeanMetric()
        preds_metric = CatMetric()
        target_metric = CatMetric()

        for subject in self.dataset.keys():
            self.logger.info(f"\n[Subject-{subject} Training ...]")
            self.logger.info("-" * 50)

            eegdata = self._process_sub_dataset(subject)
            sub_folder = self.data_folder / f"sub{subject}"
            sub_folder.mkdir(parents=True, exist_ok=False)
            acc, preds, target, subject_result = self._run_sub(eegdata, sub_folder)  # type: ignore
            result[f"subject_{subject}"] = subject_result

            acc_metric.update(acc)
            preds_metric.update(preds)
            target_metric.update(target)

            self.filer.write(f"Subject_{subject} Acc = {acc*100:.2f}%\n")

        acc = acc_metric.compute()
        self.filer.write(f"Model Acc = {acc*100:.2f}%\n")

        self.logger.info("-" * 50)
        self.logger.info(f"[Model Acc = {acc*100:.2f}%]")

        result.update(
            {
                "acc": acc,
                "preds": preds_metric.compute(),
                "target": target_metric.compute(),
            }
        )
        return result
