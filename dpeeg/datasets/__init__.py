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
from .motor_imagery import (
    BCICIV2A,
    BCICIV2B,
)
from .utils import save, load, save_dataset
