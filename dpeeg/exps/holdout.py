# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from .base import ClsExp
from ..trainer.classifier import BaseClassifier
from ..transforms.base import SplitTrainTest
from ..utils import get_init_args


class HoldOut(ClsExp):
    """Holdout cross validation experiment.

    Validate the performance of the model on unseen data using holdout cross
    validation. Only one training and testing is required, so it is fast and
    suitable for large-scale datasets and fast model evaluation.

    Parameters
    ----------
    trainer : Trainer
        Trainer used for training module on dataset.
    out_folder : str, optional
        Store all experimental results in a folder named with the model class
        name in the specified folder. Default is
        '~/dpeeg/out/model/exp/dataset/timestamp'.
    timestamp : bool
        Output folders are timestamped.
    """

    def __init__(
        self,
        trainer: BaseClassifier,
        out_folder: str | None = None,
        timestamp: bool = True,
        verbose: int | str = "INFO",
    ) -> None:
        super().__init__(
            get_init_args(self, locals(), ret_dict=True),
            trainer=trainer,
            out_folder=out_folder,
            timestamp=timestamp,
            verbose=verbose,
        )

    def _run_sub_classifier(self, eegdata, sub_folder):
        result = self.trainer.fit(
            trainset=eegdata["train"],
            testset=eegdata["test"],
            log_dir=sub_folder,
        )
        return result

    def _run_sub_classifier_two_stage(self, eegdata, sub_folder):
        split_eegdata = SplitTrainTest()(eegdata["train"])
        train_set = split_eegdata["train"]
        valid_set = split_eegdata["test"]
        test_set = eegdata["test"]

        result = self.trainer.fit(
            trainset=train_set,
            validset=valid_set,
            testset=test_set,
            log_dir=sub_folder,
        )
        return result

    def _run_sub(self, eegdata, sub_folder):
        self.logger.info(f"\n# ---------- {sub_folder.name} ---------- #")
        result = self._run_sub_func(self._trans_eegdata(eegdata), sub_folder)
        return (
            result["test"]["acc"],
            result["test"]["preds"],
            result["test"]["target"],
            result,
        )
