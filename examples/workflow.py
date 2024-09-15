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

import dpeeg
from dpeeg.datasets import BCICIV2A

dataset = BCICIV2A(tmin=0, tmax=4)

# %%
# Step 2: Initialize the Deep Learning Model
# ------------------------------------------
# 
# Here we use ``EEGNet`` as the model to be verified, or you can build your own
# deep learning model for verification. Since we use the EEG data of all 22 
# channels of the BCICIV2A dataset, and the sample time window is 4 s (the 
# total time sample points are 1000 when the sampling rate is 250 Hz), and the 
# dataset has left hand, right hand, feet, and tongue decoding characters. 
# Therefore, we set the initialization parameters of ``EEGNet`` according to 
# the characteristics of the dataset.

net = dpeeg.models.EEGNet(nCh=22, nTime=1000, cls=4)

# %%
# Step 3: Preprocessing Data
# --------------------------
# 
# Once the dataset is determined, we can preprocess it according to the 
# experimental design requirements to meet the needs of the experiment or model 
# input. Here, we assume the following steps for processing the raw data:
# 
# 1. Split the dataset according to the experimental requirements. In this case,
#    We use the validation method from the competition associated with this 
#    dataset, i.e., session one is uesd as the training set to train the model,
#    and session two is used as the test set to validate the model.
# 2. Normalize the split data to reduce noise to a certain extent.
# 3. Since we are validating the ``EEGNet`` model, we expand the eeg dimensions
#    according to the model's input requirements.
# 
# .. note::
#    The dataset is not split during initialization, so we usually need to use
#    ``SpliteTraintTest`` to divide the dataset.

from dpeeg import transforms

trans = transforms.Sequential(
    transforms.SplitTrainTest(
        train_sessions=['session_1'], 
        test_sessions=['session_2'],
    ),
    transforms.ZscoreNorm(),
    transforms.Unsqueeze(),
)

# %%
# Step 4: Select Trainer and Experiment
# -------------------------------------
# 
# We have already decided to use the BCICIV2A dataset to evaluate the 
# performance of the ``EEGNet`` model. Finally, we just need to choose the
# training method for the model and the experimental approach to assess its
# performance. The roles of the trainer and the experiment are as follows:
# 
# - The `trainer` determines the training parameters such as the model's loss
#   function, optimier, and batch size, and manages the overall training 
#   process.
# - The `experiment` is responsible for training the model for each subject in
#   the dataset using the specified trainer, and for gathering all the results
#   for further analysis.
# 
# In this tutorial, we will use a basic classification model and a simple 
# hold out experiment to quickly evaluate the model's performance on the
# corresponding dataset.
 
exp = dpeeg.exps.HoldOut(
    trainer=dpeeg.trainer.Classifier(net),
    out_folder='out'
)
result = exp.run(dataset, transforms=trans)

# %%
# .. tip::
#    It is recommended to defer transforms (``trans``) until they are passed as
#    experiment parameters rather than transforming the entire dataset upfront.
#    This is useful for transforms that would increase the size of the data and
#    thus take up a lot of memory.