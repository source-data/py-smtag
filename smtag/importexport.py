import os
from zipfile import ZipFile
import json
from datetime import datetime
import torch
from smtag.config import MODEL_DIR
from smtag.builder import Builder

def export_model(model,  opt):
    opt = opt
    suffixes = []
    suffixes.append("_".join([f for f in model.output_semantics]))
    #suffixes.append(_".join([f for f in opt['collapsed_features']]))
    suffixes.append(datetime.now().isoformat("-",timespec='minutes').replace(":", "-"))
    suffix = "_".join(suffixes)
    model_path = os.path.join(MODEL_DIR, f"{opt['namebase']}_{suffix}.sddl")
    #torch.save(model, model_filename) # does not work
    archive_path = os.path.join(MODEL_DIR, f"{opt['namebase']}_{suffix}.zip")
    option_path = os.path.join(MODEL_DIR, f"{opt['namebase']}_{suffix}.json")
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

def load_model(archive_filename):
    archive_path = os.path.join(MODEL_DIR, archive_filename)
    with ZipFile(archive_path) as myzip:
        for member in myzip.infolist():
            myzip.extract(member)
            _, ext = os.path.splitext(member.filename)
            if ext == '.sddl':
                model_path = member.filename
            elif ext == '.json':
                option_path = member.filename

    with open(option_path, 'r') as optionfile:
        opt = json.load(optionfile)
    print("trying to build model with options:")
    print(opt)
    model =  Builder(opt).model
    model.load_state_dict(torch.load(model_path))
    print(model)
    return model