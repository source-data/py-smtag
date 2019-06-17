# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import numpy as np
import torch
from torch.utils.data import Dataset
import math
import os
from typing import List
from random import shuffle
from functools import lru_cache
from ..common.mapper import Catalogue, concept2index
from ..common.progress import progress
from ..common.utils import tokenize, cd
from ..datagen.convert2th import EncodedExample
from .. import config


class Data4th(Dataset):

    def __init__(self, opt: 'Options', subdir: str):
        # use a list of data_dir_path to aggregate several training set
        data_dir_path_list = [os.path.join(config.data4th_dir, dir, subdir) for dir in opt.data_path_list]
        self.path_list = []
        for data_dir_path in data_dir_path_list:
            new_list = [os.path.join(data_dir_path, d) for d in os.listdir(data_dir_path) if d not in config.dirignore]
            self.path_list += new_list
        self.N = len(self.path_list)
        self.opt = opt
        self.opt.L = self.sniff()
        self.millefeuille = Assembler(self.opt)
        self.tokenized = []
        print(f"listed {len(self.path_list)} data packages")

    def sniff(self):
        sample_input = torch.load(os.path.join(self.path_list[0], EncodedExample.textcoded_filename))
        L = sample_input.size(2)
        return L

    def __len__(self):
        return self.N

    #@lru_cache(maxsize=config.cache_dataset)
    def __getitem__(self, i):
        path = self.path_list[i]
        textcoded = torch.load(os.path.join(path, EncodedExample.textcoded_filename)).float()
        features = torch.load(os.path.join(path, EncodedExample.features_filename)).float()
        ocr_context = None
        if self.opt.use_ocr_context and os.path.isfile(os.path.join(path, EncodedExample.ocr_context_filename)):
            ocr_context = torch.load(os.path.join(path, EncodedExample.ocr_context_filename)).float()
        viz_context = torch.Tensor(0) # empty tensor
        if self.opt.viz_context_table and os.path.isfile(os.path.join(path, EncodedExample.viz_context_filename)):
            viz_context = torch.load(os.path.join(path, EncodedExample.viz_context_filename)).float()
            viz_context = viz_context.view(1, -1) # vectorize to 1 x V; can be concatenated into batches torch.cat(vector_list, 0) and used in nn.Linear()
        with open(os.path.join(path, EncodedExample.text_filename), 'r') as f:
            text = f.read()
        with open(os.path.join(path, EncodedExample.provenance_filename), 'r') as f:
            provenance = f.read()
        encoded_example = EncodedExample(provenance, text, features, textcoded, ocr_context, viz_context)
        input, output = self.millefeuille.assemble(encoded_example)
        return (provenance, input, output, viz_context)


class Assembler:

    def __init__(self, opt: 'Options'):
        self.opt = opt

    def assemble(self, encoded_example):
        
        # INPUT: ENCODED TEXT SAMPLES
        input = encoded_example.textcoded

        # INPUT: IMAGE OCR FEATURES AS ADDITIONAL INPUT
        ocr_features = None
        if self.opt.use_ocr_context =='ocr1':
            ocr_features = encoded_example.ocr_context[ : , -2, : ] + encoded_example.ocr_context[ : , -1, : ] # fuses vertical and horizontal features
        elif self.opt.use_ocr_context =='ocr2':
            ocr_features = encoded_example.ocr_context[ : , -2: , : ] # only vertical horizontal features
        elif self.opt.use_ocr_context == 'ocrxy':
            ocr_features = encoded_example.ocr_context

        if ocr_features is not None:
            input = torch.cat((input, ocr_features), 1)

        # OUTPUT SELECTION AND COMBINATION OF FEATURES
        output = torch.cat([encoded_example.features[ : , concept2index[f], : ] for f in self.opt.selected_features], 0)
        output.unsqueeze_(0)

        # OUTPUT: add a feature for untagged characters; necessary for softmax classification
        no_tag_feature = output.sum(1) # 3D 1 x C x L, is superposition of all features so far
        no_tag_feature.unsqueeze_(0)
        no_tag_feature = 1 - no_tag_feature # sets to 1 for char not tagged and to 0 for tagged characters
        output = torch.cat((output, no_tag_feature), 1)
        return input, output
