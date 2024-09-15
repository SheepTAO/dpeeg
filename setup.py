from setuptools import setup, find_packages

__version__ = "0.4.0"
install_requires = [
    "numpy>=1.21.5",
    "einops>=0.7.0",
    "seaborn>=0.12.1",
    "pooch>=1.6.0",
    "mne>=1.6",
    "scipy>=1.11.1",
    "scikit-learn>=1.0.2",
    "tqdm>=4.64.1",
    "torchmetrics>=1.0.0",
    "torchinfo>=1.5.0",
    "tensorboard",
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dpeeg",
    version=__version__,
    author="SheepTAO",
    author_email="sheeptao@outlook.com",
    license="MIT",
    description="Deep learning with EEG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SheepTAO/dpeeg",
    packages=find_packages(),
    keywords=["eeg", "deep learning", "pytorch"],
    python_requires=">=3.10",
    install_requires=install_requires,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Operating System :: OS Independent",
    ],
)
