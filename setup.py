from dpeeg import __version__
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='dpeeg',
    version=__version__,
    author='SheepTAO',
    author_email='sheeptao@outlook.com',
    license='MIT',
    description='Deep learning with EEG',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SheepTAO/dpeeg',
    packages=find_packages(),
    platforms=['all'],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        
        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
)