# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from .base import ClsExp
from ..trainer.classifier import BaseClassifier
from ..transforms.base import SplitTrainTest


class HoldOut(ClsExp):
    def __init__(
        self,
        repr: str,
        trainer: BaseClassifier,
        out_folder: str | None = None,
        verbose: int | str = "INFO",
    ) -> None:
        super().__init__(repr, trainer, out_folder, verbose)

    def _run_sub_classifier(self, eegdata, sub_folder):
        result = self.trainer.fit(
            trainset=eegdata["train"],
            testset=eegdata["test"],
            log_dir=sub_folder,
        )

        self.logger.info(f"Acc: Train={result['train']['acc']} | Test={result['test']['acc']}")
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

        self.logger.info(
            f"Acc: Train={result['train']['acc']} | "
            f"Valid={result['valid']['acc']} | "
            f"Test={result['test']['acc']}"
        )
        return result

    def _run_sub(self, eegdata, sub_folder):
        self.logger.info(f"\n# ---------- {sub_folder.name} ---------- #")
        result = self._run_sub_func(self._trans_eegdata(eegdata), sub_folder)
        return (
            result["test"]["acc"], 
            result["test"]["preds"], 
            result["test"]["target"], 
            result
        )
