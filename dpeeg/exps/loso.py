# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from .base import ClsExp
from ..datasets.base import EEGData, SplitEEGData
from ..transforms.base import ToEEGData
from .holdout import HoldOut
from .kfold import KFold


class LOSO(ClsExp):
    def _process_sub_dataset(self, subject: int):
        """Merge eegdata according to LOSO experiment requirements."""
        self.logger.info(f"[Leave Subject {subject} Out]")

        train_egd = EEGData()
        for sub, egd in self.dataset.items():
            egd = ToEEGData()(egd, verbose=False)
            if sub == subject:
                test_egd = egd
            else:
                train_egd.append(egd)

        return SplitEEGData(train_egd, test_egd)


class LOSO_HoldOut(LOSO, HoldOut):
    """Leave-One-Subject-Out Holdout cross validation experiment."""

    pass


class LOSO_KFold(LOSO, KFold):
    """Leave-One-Subject-Out K-Fold cross validation experiment."""

    def _process_sub_dataset(self, subject: int):
        if self.isolate_testset == False:
            self.isolate_testset = True
            self.logger.warning(
                "The `isolate_testset` is ignored in LOSO_KFold experiment."
            )
        return super()._process_sub_dataset(subject)
