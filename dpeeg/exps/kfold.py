# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from sklearn.model_selection import StratifiedKFold
from torchmetrics.aggregation import MeanMetric, CatMetric

from .base import ClsExp
from ..trainer.classifier import BaseClassifier
from ..datasets.base import SplitEEGData
from ..transforms.base import ToEEGData, SplitTrainTest
from ..tools import Filer
from ..utils import DPEEG_SEED, get_init_args


class KFold(ClsExp):
    """K-Fold cross validation experiment.

    The KFold experiment divides the dataset into K non-overlapping subsets
    (i.e., "folds") and repeatedly trains and tests the model. The purpose is
    to reduce the dependence of the model evaluation results on the way the
    dataset is divided and to improve the stability and reliability of the
    evaluation results. However, its computational cost is high, especially for
    large datasets and complex models. It may take a long time to complete the
    training of all folds.

    Parameters
    ----------
    trainer : Trainer
        Trainer used for training module on dataset.
    out_folder : str, optional
        Store all experimental results in a folder named with the model class
        name in the specified folder. Default is
        '~/dpeeg/out/model/exp/dataset/timestamp'.
    k : int, optional
        k of k-Fold.
    isolate_testset : bool
        By default, the test set is independent, that is, the k-fold cross-
        validation at this time only divides the training set and the
        verification set based on the training set to implement an early
        stopping mechanism, and finally evaluates on the isolated test set.
        If False, the test set is for each fold of k-fold cross-validation.
    shuffle : bool
        Shuffle before kfold.
    seed : int
        Seed of random for review.
    timestamp : bool
        Output folders are timestamped.

    Notes
    -----
    If ``isolate_testset`` False, please provide the ``transforms`` parameter
    of the ``run`` function to avoid data leakage caused by operations such
    as data augmentation in advance.
    """

    def __init__(
        self,
        trainer: BaseClassifier,
        out_folder: str | None = None,
        k: int = 5,
        isolate_testset: bool = True,
        shuffle: bool = True,
        seed: int = DPEEG_SEED,
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
        self.k = str(k)
        self.isolate_testset = isolate_testset
        self.skf = StratifiedKFold(k, shuffle=shuffle, random_state=seed)

    def _run_sub_classifier(self, eegdata, sub_folder):
        if self.isolate_testset:
            raise TypeError("Isolated test set is useless with a `Classifier` trainer.")

        eegdata = ToEEGData()(eegdata, verbose=False)

        result = {}
        filer = Filer(sub_folder / "summary.txt")
        train_acc_metric = MeanMetric()
        test_acc_metric = MeanMetric()
        preds_metric = CatMetric()
        target_metric = CatMetric()

        for exp_idx, (train_idx, test_idx) in enumerate(
            self.skf.split(eegdata["edata"], eegdata["label"]), start=1
        ):
            self.logger.info(f"\n# ---- {sub_folder.name}_exp{exp_idx} ---- #")

            trainset = eegdata.index(train_idx)
            testset = eegdata.index(test_idx)
            egd = self._trans_eegdata(SplitEEGData(trainset, testset))
            train_set = egd["train"]
            test_set = egd["test"]

            exp_folder = sub_folder / f"exp{exp_idx}"
            exp_result = self.trainer.fit(
                trainset=train_set,
                testset=test_set,
                log_dir=exp_folder,
            )

            result[f"exp{exp_idx}"] = exp_result
            train_acc_metric.update(exp_result["train"]["acc"])
            test_acc_metric.update(exp_result["test"]["acc"])
            preds_metric.update(exp_result["test"]["preds"])
            target_metric.update(exp_result["test"]["target"])

            filer.write(
                f"Exp_{str(exp_idx).zfill(len(self.k))} Acc: "
                f"Train={exp_result['train']['acc']:.4f} | "
                f"Test={exp_result['test']['acc']:.4f}\n"
            )

        train_acc = train_acc_metric.compute()
        test_acc = test_acc_metric.compute()
        filer.write(f"Avg Acc = {test_acc*100:.2f}%\n")
        self.logger.info("-" * 30)
        self.logger.info(f"Avg Acc: Train={train_acc:.4f} | Test={test_acc:.4f}")

        result.update(
            {
                "acc": test_acc,
                "preds": preds_metric.compute(),
                "target": target_metric.compute(),
            }
        )
        return result

    def _run_sub_classifier_two_stage(self, eegdata, sub_folder):
        if self.isolate_testset:
            eegdata = self._trans_eegdata(eegdata)
            X = eegdata["train"]["edata"]
            y = eegdata["train"]["label"]
        else:
            eegdata = ToEEGData()(eegdata, verbose=False)
            X = eegdata["edata"]
            y = eegdata["label"]

        result = {}
        filer = Filer(sub_folder / "summary.txt")
        train_acc_metric = MeanMetric()
        valid_acc_metric = MeanMetric()
        test_acc_metric = MeanMetric()
        preds_metric = CatMetric()
        target_metric = CatMetric()

        for exp_idx, (train_idx, test_idx) in enumerate(self.skf.split(X, y), start=1):
            self.logger.info(f"\n# ---- {sub_folder.name}_exp{exp_idx} ---- #")

            if self.isolate_testset:
                train_set = eegdata["train"].index(train_idx)
                valid_set = eegdata["train"].index(test_idx)
                test_set = eegdata["test"]
            else:
                trainset = eegdata.index(train_idx)  # type: ignore
                testset = eegdata.index(test_idx)  # type: ignore
                egd = self._trans_eegdata(SplitEEGData(trainset, testset))

                test_set = egd["test"]
                trainset = egd["train"]
                egd = SplitTrainTest()(trainset)
                train_set = egd["train"]
                valid_set = egd["test"]

            exp_folder = sub_folder / f"exp{exp_idx}"
            exp_result = self.trainer.fit(
                trainset=train_set,
                validset=valid_set,
                testset=test_set,
                log_dir=exp_folder,
            )

            result[f"exp{exp_idx}"] = exp_result
            train_acc_metric.update(exp_result["train"]["acc"])
            valid_acc_metric.update(exp_result["valid"]["acc"])
            test_acc_metric.update(exp_result["test"]["acc"])
            preds_metric.update(exp_result["test"]["preds"])
            target_metric.update(exp_result["test"]["target"])

            filer.write(
                f"Exp_{str(exp_idx).zfill(len(self.k))} Acc: "
                f"Train={exp_result['train']['acc']:.4f} | "
                f"Valid={exp_result['train']['acc']:.4f} | "
                f"Test={exp_result['test']['acc']:.4f}\n"
            )

        train_acc = train_acc_metric.compute()
        valid_acc = valid_acc_metric.compute()
        test_acc = test_acc_metric.compute()
        filer.write(f"Avg Acc = {test_acc*100:.2f}%\n")
        self.logger.info("-" * 30)
        self.logger.info(
            f"Avg Acc: Train={train_acc:.4f} | Valid={valid_acc:.4f} | "
            f"Test={test_acc:.4f}"
        )

        result.update(
            {
                "acc": test_acc,
                "preds": preds_metric.compute(),
                "target": target_metric.compute(),
            }
        )
        return result

    def _run_sub(self, eegdata, sub_folder):
        """Basic K-Fold cross validation function.

        Returns
        -------
        Return test_acc, test_kappa, test_preds, test_target and results :\n
        results = {
            'expNo_1' : { ... },
            'expNo_2' : { ... },
            .
            .
        }
        """
        self.timer.start("kfold")
        self.logger.info(f"\n# ---------- {sub_folder.name} ---------- #")
        result = self._run_sub_func(eegdata, sub_folder)
        h, m, s = self.timer.stop("kfold")
        self.logger.info(
            f"\n[{self.k}Fold CV Finish] - [Cost Time = {h}H:{m}M:{s:.2f}S]"
        )

        return result["acc"], result["preds"], result["target"], result
