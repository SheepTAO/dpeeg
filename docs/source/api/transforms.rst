Transforms
===============

This is used to transform the data within dpeeg. The transformation is applied 
to the data using the ``__call__`` function, supporting input data types of 
:ref:`eeg_data` and :ref:`eeg_dataset`.

Usage can be referenced from the following examples:

.. automethod:: dpeeg.transforms.base.Transforms.__call__

.. important::

   When the input type is :ref:`eeg_dataset`, the transformation will be 
   applied to all subject data within the dataset. By default, all subject
   data will be read into memory for processing and returned. The 
   transformation can be made subject-wise by specifying the ``iter`` 
   parameter to save memory overhead.

.. currentmodule:: dpeeg.transforms

Core
-----------------------------

.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :template: transforms.rst

   Sequential
   SplitTrainTest
   ToEEGData

Commonly
-----------------------------

.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :template: transforms.rst

   Identity
   Crop
   SlideWin
   Unsqueeze
   Squeeze
   Transpose
   FilterBank
   ApplyFunc
   LabelMapping
   PickLabel

Normalization
-----------------------------

.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :template: transforms.rst

   ZscoreNorm
   MinMaxNorm

Data Augmentation
-----------------------------

.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :template: transforms.rst

   SegRecTime
   SlideWinAug


Compute Metrics
-----------------------------

.. autosummay::
   :toctree: ../generated
   :nosignatures:
   :template: transform.rst

   Pearsonr