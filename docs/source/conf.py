# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute.
import os
import sys

sys.path.insert(0, os.path.abspath("../../dpeeg"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from dpeeg import __version__

project = "dpeeg"
copyright = "2024, SheepTAO"
author = "SheepTAO"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "_static/banner.svg"
html_favicon = "_static/logo.svg"

# https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html

html_theme_options = {
    "navbar_align": "left",
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "header_links_before_dropdown": 4,
    "external_links": [
        {"name": "PyTorch", "url": "https://pytorch.org/"},
        {"name": "MNE", "url": "https://mne.tools/stable/index.html"},
        {"name": "MOABB", "url": "https://moabb.neurotechx.com/docs/index.html"},
        {
            "name": "TorchMetrics",
            "url": "https://lightning.ai/docs/torchmetrics/stable/",
        },
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SheepTAO/dpeeg",
            "icon": "fa-brands fa-github",
        },
    ],
}


# -- Sphinx-gallery configuration  -------------------------------------------
# https://sphinx-gallery.github.io/stable/getting_started.html#create-simple-gallery

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
}
