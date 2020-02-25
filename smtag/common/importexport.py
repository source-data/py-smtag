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
from ..train.builder import SmtagModel
from .options import Options

def export_model(model: SmtagModel, filename: str, model_dir: str=config.model_dir):
    path = os.path.join(model_dir, filename)
    model_io.export_model(model, path, filename)

def load_autoencoder(model_dir:str, filename:str):
    path = os.path.join(model_dir, filename)
    autoencoder = model_io.load_autoencoder(path)
    return autoencoder

def load_container(model_dir:str, filename:str):
    path = os.path.join(model_dir, filename)
    model = model_io.load_container(path, Container1d, CatStack1d)
    return model