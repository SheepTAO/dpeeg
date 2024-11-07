"""
Classification Model Validation
===============================

In this tutorial, we will demonstrate a basic dpeeg learning classification 
model verification process, including data reading and conversion, model 
definition, trainer initialization, and experimental verification.
"""

# %%
# Step 1: Initialize the Dataset
# ------------------------------
#
# Firstly, we import the necessary packages and set a global random seed to
# ensure the reproducibility of the results. Here, we use the
# :class:`~dpeeg.datasets.BCICIV2A` dataset built into dpeeg (which needs to be
# downloaded locally for the first time) as validation. Each sample uses 4 s of
# EEG data as input, and no baseline calibration is performed. For the sake of
# tutorial, we will only use the data from subjects 1 and 2 to validate the
# model.

import dpeeg
from dpeeg.datasets import BCICIV2A
from dpeeg import transforms

dpeeg.set_seed()

dataset = BCICIV2A(subjects=[1, 2], tmin=0, tmax=4)

# %%
# Step 2: Initialize the Deep Learning Model
# ------------------------------------------
#
# Here we use :class:`~dpeeg.models.EEGNet` as the model to be verified, or you
# can build your own deep learning model for verification. Since we use the EEG
# data of all 22 channels of the BCICIV2A dataset, and the sample time window
# is 4 s (the total time sample points are 1000 when the sampling rate is
# 250 Hz), and the dataset has left hand, right hand, feet, and tongue decoding
# characters. Therefore, we set the initialization parameters of EEGNet
# according to the characteristics of the dataset.

net = dpeeg.models.EEGNet(nCh=22, nTime=1000, nCls=4)

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
# 3. Labels may not start from 0, and therefore often need to updated.
# 4. Since we are validating the EEGNet model, we expand the eeg dimensions
#    according to the model's input requirements.
#
# .. note::
#    The dataset is not split during initialization, so we usually need to use
#    :class:`~dpeeg.transforms.SplitTraintTest` to divide the dataset.

trans = transforms.Sequential(
    transforms.SplitTrainTest(
        cross=True,
        train_sessions=["session_1"],
        test_sessions=["session_2"],
    ),
    transforms.ZscoreNorm(),
    transforms.LabelMapping(),
    transforms.Unsqueeze(),
)

# %%
# Step 4: Select Trainer and Experiment
# -------------------------------------
#
# We have already decided to use the BCICIV2A dataset to evaluate the
# performance of the EEGNet model. Finally, we just need to choose the
# training method for the model and the experimental approach to assess its
# performance. The roles of the trainer and the experiment are as follows:
#
# - The **trainer** determines the training parameters such as the model's loss
#   function, optimier, and batch size, and manages the overall training
#   process.
# - The **experiment** is responsible for training the model for each subject
#   in the dataset using the specified trainer, and for gathering all the
#   results for further analysis.
#
# In this tutorial, we will use a basic :class:`~dpeeg.trainer.Classifier`
# trainer and a simple :class:`~dpeeg.exps.HoldOut` experiment to quickly
# evaluate the model's performance on the corresponding dataset.

exp = dpeeg.exps.HoldOut(
    trainer=dpeeg.trainer.Classifier(net, max_epochs=500, no_increase_epochs=20),
    out_folder="out",
)
result = exp.run(dataset, transforms=trans)

# %%
# .. tip::
#    It is recommended to defer transforms (``trans``) until they are passed as
#    experiment parameters rather than transforming the entire dataset upfront.
#    This is useful for transforms that would increase the size of the data and
#    thus take up a lot of memory.


# %%
# Advance 1: Dataset Level Validation
# -------------------------------------
#
# The experiments that previously validated model performance were based on the
# subject level, meaning that for a dataset with multiple subjects, the model
# would train a separate model for each subject. The average performance of
# each subject's model was then used to represent the model's performance on
# the dataset, which is effective for cases where each subject has multiple
# labels (such as motor imagery, emotion recognition, etc.). However, special
# treatment is required for cases where each subject has only one label (such
# as psychiatric disease diagnosis, etc.).
#
# Here we use the :class:`~dpeeg.datasets.MODMA_128_Resting` dataset for
# tutorial. As before, the model validation consists of four steps, with the
# difference lying in the processing of the dataset. Below, we first define
# the dataset and also use EEGNet as the validation model. We use all EEG
# channels and 2 seconds of data (2 * 250 Hz = 500 time samples) as the model
# input:

dataset = dpeeg.datasets.MODMA_128_Resting(tmax=300)
net = dpeeg.models.EEGNet(nCh=128, nTime=500, nCls=2)

# %%
# Since the depression dataset consists of 5-minute EEG recordings for each
# subject, with the data labels determined by the subject's depression status,
# the dataset contains as many labels as there are subjects. To prepare the
# data for model training, we need to mix the EEG data from different subjects
# and segment the original 5-minute EEG recordings into multiple non-overlapping
# 2-second samples:

data = transforms.split_subjects(dataset)
trans = transforms.Sequential(
    transforms.SlideWin(500),
    transforms.SplitTrainTest(),
    transforms.LabelMapping(),
    transforms.Unsqueeze(),
)
data = trans(data)

# %%
# Since :class:`~dpeeg.transforms.split_subjects` will change the entire
# dataset structure, it needs to be used separately.
#
# Finally, define the trainer and experiment to validate the model:

trainer = dpeeg.trainer.Classifier(net, nGPU=1)
exp = dpeeg.exps.HoldOut(trainer, out_folder="out")
result = exp.run(data, dataset_name="MODMA_128_Resting")
