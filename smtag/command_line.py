# -*- coding: utf-8 -*-
# Command Line Interface
# http://python-packaging.readthedocs.io/en/latest/command-line-scripts.html

from .train   import meta       as train_meta
from .train   import evaluator  as train_eval
from .datagen import neo2xml    as datagen_neo2xml
from .datagen import convert2th as datagen_convert2th
from .predict import engine     as predict_engine

def meta():
    return train_meta.main()

def neo2xml():
    return datagen_neo2xml.main()

def convert2th():
    return datagen_convert2th.main()

def predict():
    return predict_engine.main()

def eval():
    return train_eval.main()

def about():
    return """
SmartTag command line toolchain (2018 EMBO)
Available commands:

    - smtag: this help command
    - smtag-meta: Training of new models is managed via `meta`. Run the help command to get a list of options.
    - smtag-neo2xml: generates xml files from sd-graph
    - smtag-convert2th: encode and sample examples provided as xml files or brat files
    - smtag-predict: command line interface to the SmartTag taggin engine
    """
