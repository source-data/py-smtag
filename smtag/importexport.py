# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import os
from zipfile import ZipFile
import json
from datetime import datetime
import torch
from smtag.config import MODEL_DIR
from smtag.builder import build
from smtag.utils import cd

def export_model(model, custom_name = '', model_dir = MODEL_DIR):
    opt = model.opt
    if custom_name:
        name = custom_name
    else:
        suffixes = []
        suffixes.append("_".join([f for f in model.output_semantics]))
        #suffixes.append(_".join([f for f in opt['collapsed_features']]))
        suffixes.append(datetime.now().isoformat("-",timespec='minutes').replace(":", "-"))
        suffix = "_".join(suffixes)
        name = f"{opt['namebase']}_{suffix}"
    model_path = f"{name}.sddl" #os.path.join(name, f"{name}.sddl")
    #torch.save(model, model_filename) # does not work
    archive_path = f"{name}.zip" #os.path.join(model_dir, f"{name}.zip")
    option_path = f"{name}.json" # os.path.join(model_dir, f"{name}_{suffix}.json")
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
            print(f"saved {info.filename} (size: {info.file_size})")
        return myzip

def load_model(archive_filename, model_dir=MODEL_DIR):
    archive_path = archive_filename # os.path.join(model_dir, archive_filename)
    with cd(model_dir):
        with ZipFile(archive_path) as myzip:
            print(f"Extracting:")
            myzip.printdir()
            myzip.extractall()
            for filename in myzip.namelist():
                _, ext = os.path.splitext(filename)
                #_, filename = os.path.split(filename)
                if ext == '.sddl':
                    model_path = filename
                    print(f"extracted {model_path}")
                elif ext == '.json':
                    option_path = filename
                    print(f"extracted {option_path}")

        with open(option_path, 'r') as optionfile:
            opt = json.load(optionfile)
        print("trying to build model with options:")
        print(opt)
        model =  build(opt)
        model.load_state_dict(torch.load(model_path))
        print(f"removing {model_path}")
        os.remove(model_path)
        print(f"removing {option_path}")
        os.remove(option_path)
    return model