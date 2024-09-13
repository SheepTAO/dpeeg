Transforms
==========

.. currentmodule:: dpeeg.transforms

Core Transformer
-----------------------------

.. autosummary::
    :toctree: ../generated
    :template: transforms.rst

    Sequential
    SplitTrainTest
    ToEEGData

Commonly Transformer
-----------------------------

.. autosummary::
    :toctree: ../generated
    :template: transforms.rst

    Identity
    Crop
    SlideWin
    Unsqueeze
    Squeeze
    FilterBank
    ApplyFunc
    LabelMapping
    PickLabel

Normalization Transformer
-----------------------------

.. autosummary::
    :toctree: ../generated
    :template: transforms.rst

    ZscoreNorm
    MinMaxNorm

Data Augmentation Transformer
-----------------------------

.. autosummary::
    :toctree: ../generated
    :template: transforms.rst

    SegRecTime
    SlideWinAug
