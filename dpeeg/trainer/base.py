# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from abc import ABC, abstractmethod
from pathlib import Path

from ..tools.logger import Logger
from ..tools.timer import Timer
from ..utils import _format_log


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

    def __init__(self, repr: dict, model, verbose) -> None:
        super().__init__()
        self._repr = repr
        self.model = model
        self.verbose = verbose
        self.controller = None
        self.logger = Logger("dpeeg_trainer", clevel=verbose)
        self.timer = Timer()

    def _reset_logger(self, log_dir: str | Path):
        """Reset the logger to set the log storage location for different
        experiments, usually called in the ``exps.fit`` method.
        """
        log_dir = Path(log_dir).absolute()
        self.logger = Logger(
            str(log_dir),
            path=log_dir / "running.log",
            flevel=self.verbose,
            clevel=self.verbose,
        )

    def _set_controller(self, controller):
        """Specify an experiment controller for the trainer to obtain the
        experiment context. This is usually set by the experiment controller,
        so users can typically ignore it.
        """
        self.controller = controller

    def get_controller_buffer(self, name: str):
        """Get the current experiment controller. Different controllers contain
        varying context information; please refer to the :doc:`./api/exps` for
        details.

        Parameters
        ----------
        name : str
            The name of the buffer to be obtained. The buffer is usually a
            dictionary that stores the data of the experiment controller.
        """
        if self.controller is None:
            raise RuntimeError(
                "Experiment controller not found. "
                "Please use the appropriate experiment to set the controller."
            )
        return self.controller.access_buffer(name)

    @abstractmethod
    def fit(*args, **kwargs) -> dict:
        """The top-level fitting function can be overridden by different
        trainers as needed and returns the fitting results in a dictionary.
        """
        pass

    def __repr__(self) -> str:
        return _format_log(self._repr)
