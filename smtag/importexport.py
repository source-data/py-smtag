# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import os
from zipfile import ZipFile
import json
from datetime import datetime
import torch
from smtag.config import MODEL_DIR
from smtag.builder import SmtagModel
from smtag.utils import cd

def export_model(model, custom_name = '', model_dir = MODEL_DIR):
    model.cpu() # model.cpu().double() ?
    # extract the SmtagModel from the nn.DataParallel table, if necessary
    if isinstance(model, torch.nn.DataParallel):
        #print("getting SmtagModel")
        model = [m for m in model.children if isinstance(m, SmtagModel)][0]
    opt = model.opt
    if custom_name:
        name = custom_name
    else:
        suffixes = []
        suffixes.append("_".join([str(f) for f in model.output_semantics]))
        suffixes.append("_v_".join([f for f in opt['collapsed_features']]))
        suffixes.append("_&_".join([f for f in opt['overlap_features']]))
        suffixes.append(datetime.now().isoformat("-",timespec='minutes').replace(":", "-"))
        suffix = "_".join(suffixes)
        name = "{}_{}".format(opt['namebase'], suffix)
    model_path = "{}.sddl".format(name)
    #torch.save(model, model_filename) # does not work
    archive_path = "{}.zip".format(name)
    option_path = "{}.json".format(name)
    with cd(MODEL_DIR):
        with ZipFile(archive_path, 'w') as myzip:
            torch.save(model.state_dict(), model_path)
            myzip.write(model_path)
            os.remove(model_path)
            with open(option_path, 'w') as jsonfile:
                json.dump(opt, jsonfile)
            myzip.write(option_path)
            os.remove(option_path)
        for info in myzip.infolist():
            print("saved {} (size: {})".format(info.filename, info.file_size))
        return myzip

def load_model(archive_filename, model_dir=MODEL_DIR):
    archive_path = archive_filename # os.path.join(model_dir, archive_filename)
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
                    #print("extracted {}".format(model_path))
                elif ext == '.json':
                    option_path = filename
                    #print("extracted {}".format(option_path))

        with open(option_path, 'r') as optionfile:
            opt = json.load(optionfile)
        #print("trying to build model with options:")
        #print(opt)
        model =  SmtagModel(opt)
        model.load_state_dict(torch.load(model_path))
        #print("removing {}".format(model_path))
        os.remove(model_path)
        #print("removing {}".format(option_path))
        os.remove(option_path)
    return model