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
from toolbox.models import Autoencoder1d, Container1d, CatStack1d
from .. import config
from ..train.model import SmtagModel, HyperparemetersSmtagModel


def load_smtag_model(filename, model_dir:str = config.model_dir):
    path = os.path.join(model_dir, filename)
    model = model_io.load_model_from_class(path, SmtagModel)
    return model

def export_smtag_model(model: SmtagModel, filename: str, model_dir: str=config.model_dir) -> str:
    archive_path = model_io.export_model(model, model_dir, filename)
    return archive_path

def load_autoencoder(model_dir:str, filename:str):
    path = os.path.join(model_dir, filename)
    autoencoder = model_io.load_autoencoder(path)
    return autoencoder
