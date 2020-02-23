# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import os
from zipfile import ZipFile
from copy import deepcopy
import pickle
import json
from datetime import datetime
import torch
from toolbox import importexport as model_io
from toolbox.models import Container1d, CatStack1d
from .. import config
from ..train.builder import SmtagModel
from .options import Options

def export_model(model: SmtagModel, filename: str):
    path = os.path.join(config.model_dir, custom_name)
    model_io.export_model(model, path, filename)

def load_model(filename:str):
    path = os.path.join(config.model_dir, filename)
    model = model_io.load_model(path, Container1d, CatStack1d)
    return model
