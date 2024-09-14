Transforms
===============

This is used to transform the data within dpeeg. The transformation is applied 
to the data using the ``__call__`` function, supporting input data types of 
:ref:`eeg_data` and :ref:`eeg_dataset`.

Usage can be referenced from the following examples:

.. code-block:: py

    # Allows transformation of data of type eegdata:
    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> transforms.Unsqueeze()(eegdata, verbose=False)
    [edata=(16, 1, 3, 10), label=(16,)]

.. code-block:: py

    # Also allows input type eegdataset:
    >>> eegdataset = dpeeg.datasets.EEGDataset([
    ...     eegdata.copy(), eegdata.copy(), eegdata.copy()
    ... ])
    >>> transforms.Squeeze()(eegdataset, iter=False, verbose=False)
    [EEGDataset:
        [eegdataset]: Subjects=3, type=EEGData
        [event_id]: None
    ]
    
    # setting ``iter`` can iteratively obtain the eegdata of each subject
    >>> tran = transforms.Unsqueeze()
    >>> for subject, eegdata in tran(eegdataset, iter=True, verbose=False):
    ...     print(subject, eegdata)
    0 [edata=(16, 1, 3, 10), label=(16,)]
    1 [edata=(16, 1, 3, 10), label=(16,)]
    2 [edata=(16, 1, 3, 10), label=(16,)]

.. important::

    When the input type is :ref:`eeg_dataset`, the transformation will be 
    applied to all subject data within the dataset. By default, all subject
    data will be read into memory for processing and returned. The 
    transformation can be made subject-wise by specifying the ``iter`` 
    parameter to save memory overhead.

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
