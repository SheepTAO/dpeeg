# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from .base import ClsExp
from ..datasets.base import BaseDataset, EEGData, SplitEEGData
from ..transforms.base import ToEEGData
from .holdout import HoldOut
from .kfold import KFold


class LOSO(ClsExp):
    """Leave-One-Subject-Out cross validation experiment.

    Suppose that a dataset has :math:`N_s` subjects. In the LOSO_HoldOut
    experiment, the data of each subject was used as once as the test set, and
    the data of the remaining :math:`N_s - 1` subjects formed the training set.
    The average classification accuracy of the model was then calculated from
    the :math:`N_s` analyses.
    """

    _is_eeg_dataset = True  # Input data type must be a eeg dataset

    def _process_sub_dataset(self, dataset: BaseDataset, subject: int):
        """Merge eegdata according to LOSO experiment requirements."""
        if len(dataset) < 2:
            raise RuntimeError("Dataset has at least 2 subjects in LOSO.")

        self.logger.info(f"[Leave Subject {subject} Out]")
        train_egd = EEGData()
        for sub, egd in dataset.items():
            egd = ToEEGData()(egd, verbose=False)
            if sub == subject:
                test_egd = egd
            else:
                train_egd.append(egd)

        return SplitEEGData(train_egd, test_egd)


class LOSO_HoldOut(HoldOut, LOSO):
    r"""Leave-One-Subject-Out Holdout cross validation experiment.

    See Also
    --------
    :class:`.HoldOut`
    """

    pass


class LOSO_KFold(KFold, LOSO):
    """Leave-One-Subject-Out K-Fold cross validation experiment.

    See Also
    --------
    :class:`.KFold`
    """

    def _process_sub_dataset(self, dataset: BaseDataset, subject: int):
        if self.isolate_testset == False:
            self.isolate_testset = True
            self.logger.warning(
                "The `isolate_testset` is ignored in LOSO_KFold experiment."
            )
        return super()._process_sub_dataset(dataset, subject)
