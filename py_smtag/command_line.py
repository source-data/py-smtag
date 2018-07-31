# -*- coding: utf-8 -*-
# Command Line Interface
# http://python-packaging.readthedocs.io/en/latest/command-line-scripts.html
# print(__name__)
# print(__package__)
from .train import meta as train_meta
def meta():
    return train_meta.main()

def about():
    return """
SmartTag command line toolchain (2018 EMBO)
Available commands:

    - smtag: this help command
    - smtag-meta: Training of new models is managed via `meta`. Run the help command to get a list of options.
    """
