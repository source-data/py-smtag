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


def export_smtag_model(model: SmtagModel, filename: str, model_dir: str=config.model_dir):
    path = None
    try:
        if torch.cuda.is_available():
            model = deepcopy(model.module) # model.module.state_dict() the use map_location='cpu' or 'cuda:0' appropriately?
            model.cpu()
        prefix = datetime.now().isoformat("-", timespec='minutes').replace(":", "-") 
        archive_path = os.path.join(path, prefix + '_' + filename + '.zip')
        model_path = os.path.join(path, prefix + '_' + filename + '_model.th')
        opt_path = os.path.join(path, prefix + '_' + filename + '_opt.pickle')
        torch.save(model.state_dict(), model_path)
        with open(opt_path, 'wb') as f:
            pickle.dump(model.opt, f)
        with ZipFile(archive_path, 'w') as myzip:
            myzip.write(model_path)
            os.remove(model_path)
            myzip.write(opt_path)
            os.remove(opt_path)
    except Exception as e: 
        print(f"MODEL NOT SAVED: {filename}, {model_path}")
        print(e)
    return archive_path

def load_smtag_model(filename, model_dir:str = config.model_dir):
    path = os.path.join(model_dir, filename)
    print(f"\n\nloading {path}\n\n")
    opt_path, model_path = model_io.unzip_model(path)
    with open(opt_path, 'rb') as opt_file:
        opt = pickle.load(opt_file)
    print(f"trying to build model ({path} with with options:")
    print(opt)
    model =  SmtagModel(opt)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    os.remove(model_path)
    os.remove(opt_path)
    return model

# def export_model(model: SmtagModel, filename: str, model_dir: str=config.model_dir):
#     model_io.export_model(model, model_dir, filename)

def load_autoencoder(model_dir:str, filename:str):
    path = os.path.join(model_dir, filename)
    autoencoder = model_io.load_autoencoder(path)
    return autoencoder

# def load_container(model_dir:str, filename:str):
#     path = os.path.join(model_dir, filename)
#     model = model_io.load_container(path, Container1d, CatStack1d)
#     return model

