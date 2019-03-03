# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import numpy as np
import torch
from torch.utils.data import Dataset
import math
import os
from typing import List
from random import shuffle
from ..common.mapper import Catalogue, concept2index
from ..common.progress import progress
from ..common.utils import tokenize, cd
from ..datagen.convert2th import EncodedExample
from .. import config

# import logging.config
# logging.config.fileConfig('logging.conf')
# logger = logging.getLogger(__name__)

class Data4th(Dataset):

    def __init__(self, data_dir_path: str, opt: 'Options'):
        self.data_dir_path = data_dir_path
        self.path_list = [os.path.join(self.data_dir_path, d) for d in os.listdir(self.data_dir_path) if d not in config.dirignore]
        self.N = len(self.path_list)
        self.opt = opt
        self.opt.L = self.sniff()
        self.millefeuille = Assembler(self.opt)
        self.tokenized = []
        print(f"{self.data_dir_path} has {len(self.path_list)} data packages")

    def sniff(self):
        sample_input = torch.load(os.path.join(self.path_list[0], EncodedExample.textcoded_filename))
        L = sample_input.size(2)
        return L

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        path = self.path_list[i]
        textcoded = torch.load(os.path.join(path, EncodedExample.textcoded_filename)).float()
        output = torch.load(os.path.join(path, EncodedExample.features_filename)).float()
        ocr_context = None
        if os.path.isfile(os.path.join(path, EncodedExample.ocr_context_filename)):
            ocr_context = torch.load(os.path.join(path, EncodedExample.ocr_context_filename)).float()
        viz_context = None
        if os.path.isfile(os.path.join(path, EncodedExample.viz_context_filename)):
            viz_context = torch.load(os.path.join(path, EncodedExample.viz_context_filename)).float()
        with open(os.path.join(path, EncodedExample.text_filename), 'r') as f:
            text = f.read()
        with open(os.path.join(path, EncodedExample.provenance_filename), 'r') as f:
            provenance = f.read()
        encoded_example = EncodedExample(provenance, text, output, textcoded, ocr_context, viz_context)
        input, output = self.millefeuille.assemble(encoded_example)
        return (provenance, input, output)


class Assembler:

    def __init__(self, opt: 'Options'):
        self.opt = opt

    def assemble(self, encoded_example):
        """
        Processes a raw dataset into a ready-to-go dataset by converting text into tensor and assembling selected combinations of features into output features.
        Args:
            file_basename (str): the basename of the text, provenance and feature files to be uploaded and prepared.

        Returns:
            datasets (array of Dataset): the train/validation Dataset objects or the test Dataset that can be processed further by smtag.
        """
        
        # the whole thing is done with 3D 1 x C x L Tensor as a convention to simplify concatenation into batches
        input = torch.Tensor(1, self.opt.nf_input, self.opt.L)
        output = torch.Tensor(1, self.opt.nf_output, self.opt.L)
        if encoded_example.viz_context is not None:
            viz_context = encoded_example.viz_context #  N X C
            viz_context.unsqueeze_(2) # N x C x 1
            viz_context = viz_context.repeat(1, 1, self.opt.L) # N x C x L

        # INPUT: ENCODED TEXT SAMPLES
        input[ : , :config.nbits , : ] = encoded_example.textcoded
        supp_input = config.nbits

        # INPUT: IMAGE OCR FEATURES AS ADDITIONAL INPUT
        if self.opt.use_ocr_context =='ocr1':
            input[ :, supp_input:supp_input+self.opt.nf_ocr_context, : ] = encoded_example.ocr_context[ : , -2, : ] + encoded_example.ocr_context[ : , -1, : ] #### fuse vertical and horizontal features
            supp_input += self.opt.nf_ocr_context
        elif self.opt.use_ocr_context =='ocr2':
            input[ : , supp_input:supp_input+self.opt.nf_ocr_context, : ] = encoded_example.ocr_context[ : , -2: , : ] #### only vertical horizontal features
            supp_input += self.opt.nf_ocr_context
        elif self.opt.use_ocr_context == 'ocrxy':
            input[ : , supp_input:supp_input+self.opt.nf_ocr_context, : ] = encoded_example.ocr_context
            supp_input += self.opt.nf_ocr_context

        # INPUT: IMAGE VISUAL CONTEXT FEATURES AS ADDITIONAL INPUT
        if self.opt.use_viz_context:
            input[ : , supp_input:supp_input+self.opt.nf_viz_context, : ] = encoded_example.viz_context
            supp_input += self.opt.nf_viz_context

        # OUTPUT SELECTION AND COMBINATION OF FEATURES
        for j, f in enumerate(self.opt.selected_features):
            output[ : , j, : ] = encoded_example.features[ : , concept2index[f], : ]

        # OUTPUT: add a feature for untagged characters
        if self.opt.index_of_notag_class:
            no_tag_feature = output[ : , :self.opt.index_of_notag_class, :].sum(1) # 3D 1 x C x L, is superposition of all features so far
            no_tag_feature = no_tag_feature == 0 # set to 1 for char not tagged and to 0 for tagged characters
            output[ : , self.opt.index_of_notag_class, : ] = no_tag_feature.unsqueeze(0).unsqueeze(0)

        return input, output
