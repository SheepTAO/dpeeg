__version__ = "0.4.0"

from .datasets import EEGData, MultiSessEEGData, SplitEEGData, save, load
from . import transforms, exps, models, trainer, tools
from .utils import set_log_level, set_seed

# :wq
