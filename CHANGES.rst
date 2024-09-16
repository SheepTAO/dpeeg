Changelog
=========

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