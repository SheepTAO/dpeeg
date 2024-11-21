# The code for downloading datasets is adapted from the MOABB project (licensed
# under the BSD 3-Clause License). Modifications were made to align with the
# data processing methods used in the dpeeg project.

from .base import (
    EEGData,
    MultiSessEEGData,
    SplitEEGData,
    EEGDataset,
)
from .loaddataset import LoadDataset
from .utils import save, load, save_dataset
from .bnci import BCICIV2A, BCICIV2B
from .gigadb import OpenBMI_MI
from .gin import HighGamma
from .physionet import PhysioNet_MI
from .ukds import MODMA_128_Resting
from .zenodo import Ofner2017
from .harvard import MI_2
