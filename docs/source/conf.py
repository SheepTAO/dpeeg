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
from datetime import datetime

from dpeeg import __version__


sys.path.insert(0, os.path.abspath("../../dpeeg"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "dpeeg"
year = datetime.now().year
copyright = f"2023-{year}, SheepTAO"
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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]
html_css_files = ["style.css"]

html_logo = "_static/banner.svg"
html_favicon = "_static/logo.svg"

html_theme = "pydata_sphinx_theme"

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
        {"name": "Captum", "url": "https://captum.ai/"},
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SheepTAO/dpeeg",
            "icon": "fa-brands fa-github",
        },
    ],
}

# -- Sphinx-autodoc configuration  -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_member_order = "bysource"

# -- Sphinx-gallery configuration  -------------------------------------------
# https://sphinx-gallery.github.io/stable/getting_started.html#create-simple-gallery

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["auto_examples"],
    "default_thumb_file": "source/_static/thumb.png",
}

# -- Intersphinx configuration -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
