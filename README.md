# dpeeg

Dpeeg provides a complete workflow for deep learning decoding EEG tasks, including basic datasets (datasets 
can be easily customized), basic network models, model training, rich experiments, and detailed experimental 
result storage.

Each module in dpeeg is decoupled as much as possible to facilitate the use of separate modules. 

# Usage

Installation dependencies are not written yet, please install as follows:

1. Create a new virtual environment named "dpeeg" with python3.10 using Anaconda3 and activate it：

```Shell
conda create --name dpeeg python=3.10
conda activate dpeeg
```

2. Install environment dependencies

```Shell
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c anaconda ipython ipykernel pandas scikit-learn
conda install -c conda-forge torchinfo mne-base tensorboard torchmetrics seaborn
pip install moabb==0.5.0
pip install dpeeg
``` 