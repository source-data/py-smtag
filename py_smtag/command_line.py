# -*- coding: utf-8 -*-
# Command Line Interface
# http://python-packaging.readthedocs.io/en/latest/command-line-scripts.html

from .train   import meta       as train_meta
from .datagen import sdgraph2th as datagen_sdgraph2th
from .predict import engine     as predict_engine

def meta():
    return train_meta.main()

def graph2th():
    return datagen_sdgraph2th.main()

def predict():
    return predict_engine.main()

def about():
    return """
SmartTag command line toolchain (2018 EMBO)
Available commands:

    - smtag: this help command
    - smtag-meta: Training of new models is managed via `meta`. Run the help command to get a list of options.
    - smtag-train
    - smtag-graph2th
    """
