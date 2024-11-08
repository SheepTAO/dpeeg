Changelog
=========

v0.4.3
----------------

**Project maintenance**

- Add ``register_raw_hook`` method to facilitate preprocessing using ``mne``

**Documentation**

- Fix some document errors


v0.4.2
----------------

**Project maintenance**

- Fix the incomplete output information of :doc:`./api/transforms` terminal
- Normalization allows parameters calculated on the training set to be directly
  applied to the test set
- Optimize :doc:`./api/trainer`, add log manager and timer to default trainer
- Add new dataset :class:`~dpeeg.datasets.MODMA_128_Resting`
- Add new augmentation :class:`~dpeeg.transforms.GaussTime`
- :doc:`./api/exps` support for verification of :ref:`eeg_data`

**Documentation**

- Supplement the cross-references in the document to improve readability


v0.4.1
----------------

**Project maintenance**

- Optimize the runtime logic of :doc:`./api/transforms`
- Add new datasets :class:`~dpeeg.datasets.HighGamma` and 
  :class:`~dpeeg.datasets.PhysioNet_MI`
- Modify the internal dataset reading logic

**Documentation**

- Contents newly added in update version 0.4.1
- Refine the k-fold cross-validation experiment instruction document
- Use ``mne`` document decorator to reduce duplication


v0.4.0
----------------

**Documentation**

- The documentation engine has been changed from MkDocs to Sphinx
- Add dpeeg usage tutorial
- Improve API reference documentation
- Design dpeeg logo

**Project maintenance**

- Designed new :ref:`eeg_data` class to define all EEG data types
- Restructured the way to obtain the dataset, and further defined the basic the
  basic dataset and local file dataset. For details, refer to :ref:`eeg_dataset`
- According to the newly defined EEG Data, the trainer and corresponding
  experimental methods were reconstructed
- Refactored :doc:`./api/transforms` to support both EEG Data and EEG Dataset