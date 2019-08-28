# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import os
from zipfile import ZipFile
from copy import deepcopy
import pickle
import json
from datetime import datetime
import torch
from .. import config
from ..train.builder import SmtagModel
from .utils import cd, timer
from .options import Options

def file_with_suffix(namebase, selected_features):
    suffixes = []
    suffixes.append("_".join([str(f) for f in selected_features]))
    suffixes.append(datetime.now().isoformat("-",timespec='minutes').replace(":", "-"))
    suffix = "_".join(filter(None,suffixes))
    name = f"{namebase}_{suffix}".format(namebase, suffix)
    return name

@timer
def export_model(model, custom_name = '', model_dir = config.model_dir):
    # make copy of model first for continuous saving; need to leave model on GPU!
    # extract the SmtagModel from the nn.DataParallel table, if necessary
    if isinstance(model, torch.nn.DataParallel):
        internal_model = [m for m in model.children() if isinstance(m, SmtagModel)][0]
        model_copy = deepcopy(internal_model)
    else:
        model_copy = deepcopy(model)
    model_copy.cpu() # move the copy to the CPU
    state_dict = model_copy.state_dict() # the parameters to be saved
    opt = model_copy.opt
    if custom_name:
        name = custom_name
    else:
        name = file_with_suffix(opt.namebase, opt.selected_features)
    model_path = "{}.sddl".format(name)
    archive_path = "{}.zip".format(name)
    option_path = "{}.pickle".format(name)
    os.makedirs(model_dir, exist_ok=True)
    with cd(model_dir):
        with ZipFile(archive_path, 'w') as myzip:
            torch.save(state_dict, model_path)
            myzip.write(model_path)
            os.remove(model_path)
            with open(option_path, 'wb') as f:
                pickle.dump(opt, f)
            myzip.write(option_path)
            os.remove(option_path)
        for info in myzip.infolist():
            print("saved {} (size: {})".format(info.filename, info.file_size))
    return model_copy

def load_model(archive_filename, model_dir=config.model_dir, model_class=SmtagModel):
    archive_path = archive_filename # os.path.join(model_dir, archive_filename)
    print(f"\n\nloading {archive_filename} \nfrom {model_dir}\n\n")
    with cd(model_dir):
        # print("now in {}".format(os.getcwd()))
        with ZipFile(archive_path) as myzip:
            #print(f"Extracting:")
            #myzip.printdir()
            myzip.extractall()
            for filename in myzip.namelist():
                _, ext = os.path.splitext(filename)
                if ext == '.sddl':
                    model_path = filename
                elif ext == '.pickle':
                    option_path = filename

        with open(option_path, 'rb') as optionfile:
            opt = pickle.load(optionfile)
        print("trying to build model with options:")
        print(opt)
        model =  model_class(opt)
        model.load_state_dict(torch.load(model_path))
        os.remove(model_path)
        os.remove(option_path)
    return model
