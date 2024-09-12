# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from abc import ABC, abstractmethod


class Trainer(ABC):
    """Top-level trainer.

    Rewrite the trainer `fit` function according to different training methods.

    Parameters
    ----------
    model : Any
        The model that needs to be trained.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    @abstractmethod
    def fit(*args, **kwargs) -> dict:
        """The top-level fitting function can be overridden by different
        trainers as needed and returns the fitting results in a dictionary.
        """
        pass
