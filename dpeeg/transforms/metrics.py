# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from typing import Literal

import numpy as np

from ..datasets.base import EEGData
from .base import TransformsEGD
from ..utils import get_init_args


class Pearsonr(TransformsEGD):
    """Calculate the Pearson product-moment correlation coefficient at the
    electrode channel level.

    By default, calculations are performed on ``edata`` with a shape of (
    `sample`, ..., `channel`, `time`). All data in non-`channel` dimensions
    will be transferred to the `time` dimension and flattened as values of a
    single `channel`. Finally, the Pearson product-moment correlation
    coefficient between channels is calculated and added to eegdata as a
    metric. The relationship between the correlation coefficient matrix, `R`,
    and the covariance matrix, `C`, is

    .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} C_{jj} } }

    The values of `R` are between -1 and 1, inclusive.

    Parameters
    ----------
    prefix : str, optinoal
        Add a prefix to the metric key.
    mode : Literal["sample", "class", "all"]
        Defines sample-based calculations. Should be one of the following:

        - ``all``: Calculates the PCC value of all samples between channels
        - ``class``: Calculates the PCC value of samples of the same class
          between channels
        - ``sample``: Calculates the PCC value of each sample between channels

    Examples
    --------
    Calculate the PCC value between all samples of the channel by specifying
    the ``mode`` value:

    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(8, 9, 1, 3, 10),
    ...                         label=np.random.randint(0, 2, 8))
    >>> transforms.Pearsonr(prefix="p", mode="all")(eegdata, verbose=False)
    [edata=(8, 9, 1, 3, 10), label=(8,), p_pcc_a=(3, 3)]

    or calculation based on class:

    >>> transforms.Pearsonr(mode="class")(eegdata, verbose=False)
    [edata=(8, 9, 1, 3, 10), label=(8,), p_pcc_a=(3, 3), pcc_c0=(3, 3), pcc_c1=(3, 3)]

    or calculation based on sample:

    >>> transforms.Pearsonr(mode="sample")(eegdata, verbose=False)
    [edata=(8, 9, 1, 3, 10), label=(8,), p_pcc_a=(3, 3), pcc_c0=(3, 3), pcc_c1=(3, 3), pcc_s=(8, 3, 3)]
    """

    def __init__(
        self,
        prefix: str | None = None,
        mode: Literal["sample", "class", "all"] = "sample",
    ):
        super().__init__(get_init_args(self, locals(), "rp"))
        self.key = "pcc" if prefix is None else f"{prefix}_pcc"
        self.mode = mode

    def _apply_egd(self, egd: EEGData, key: str | None):
        sp, *_, nc, _ = egd["edata"].shape
        eeg = egd["edata"].swapaxes(1, -2).reshape(sp, nc, -1)

        if self.mode == "all":
            eeg = eeg.swapaxes(0, 1).reshape(nc, -1)
            egd[f"{self.key}_a"] = np.corrcoef(eeg)
        elif self.mode == "sample":
            egd[f"{self.key}_s"] = np.array([np.corrcoef(e) for e in eeg])
        elif self.mode == "class":
            for cls in egd.cls:
                cls_idx = np.where(cls == egd["label"])
                cls_eeg = eeg[cls_idx].swapaxes(0, 1).reshape(nc, -1)
                egd[f"{self.key}_c{cls}"] = np.corrcoef(cls_eeg)
        else:
            raise ValueError(
                "Parameter mode only support `sample`, `class`, and `all`, "
                f"but got `{self.mode}`."
            )
