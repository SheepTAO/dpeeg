# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from abc import ABC, abstractmethod
from pathlib import Path

from ..tools.logger import Logger
from ..tools.timer import Timer


class Trainer(ABC):
    """Top-level trainer.

    Rewrite the trainer ``fit`` function according to different training
    methods.

    Attributes
    ----------
    logger : Logger
        The log manager of the trainer. The log manager is updated by calling
        ``_reset_logger`` function when the log file output location needs to
        be updated.
    timer : Timer
        The default timer for the trainer. Other timers can be registered and
        used as needed.
    """

    def __init__(self, model, verbose) -> None:
        super().__init__()
        self.model = model
        self.verbose = verbose
        self.logger = Logger("dpeeg_trainer", clevel=verbose)
        self.timer = Timer()

    def _reset_logger(self, log_dir: str | Path):
        log_dir = Path(log_dir).absolute()
        self.logger = Logger(
            str(log_dir),
            path=log_dir / "running.log",
            flevel=self.verbose,
            clevel=self.verbose,
        )

    @abstractmethod
    def fit(*args, **kwargs) -> dict:
        """The top-level fitting function can be overridden by different
        trainers as needed and returns the fitting results in a dictionary.
        """
        pass
