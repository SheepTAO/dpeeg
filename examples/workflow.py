"""
Model Validation using dpeeg
=======================================

In this tutorial, we will demonstrate a basic dpeeg learning model verification
process, including data reading and conversion, model definition, trainer
initialization, and experimental verification.
"""

# %%
# Step 1: Initialize the Dataset
# ------------------------------
#
# We use the BCICIV2A dataset built into dpeeg (which needs to be downloaded
# locally for the first time) as validation. Each sample uses 4 s of EEG data
# as input, and no baseline calibration is performed.

from dpeeg.datasets import BCICIV2A

dataset = BCICIV2A(tmin=0, tmax=4)
