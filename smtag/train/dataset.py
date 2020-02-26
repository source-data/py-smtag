# © Thomas Lemberger 2020

import math
import os
from typing import List, Tuple, NewType
from random import shuffle
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import Dataset
from ..common.mapper import Catalogue, concept2index
from ..datagen.convert2th import EncodedExample
from ..common.progress import progress
from ..common.utils import tokenize, cd
from .. import config

# symbolic class for 3D Tensor in the Batch x Channel x Length format
BxCxL = NewType('BxCxL', torch.Tensor)

# symbolic class for 2D ByteTensors in Batch x Length format
BxL = NewType('BxL', torch.Tensor)

# items return by __get_item__() from Datasets
Item = namedtuple('Item', ['text', 'provenance', 'input', 'output', 'target_class'])

# minibatches returned after collating a list of Item()
Minibatch = namedtuple('Minibatch', ['text', 'provenance', 'input', 'output', 'target_class'])


class Data4th(Dataset):

    def __init__(self, opt: 'Options', subdir_list: List[str]):
        # use a list of data_dir_path to aggregate several training sets
        data_dir_path_list = []
        for subdir in subdir_list:
            data_dir_path_list += [os.path.join(config.data4th_dir, dir, subdir) for dir in opt.data_path_list]
        self.path_list = []
        for data_dir_path in data_dir_path_list:
            new_list = [os.path.join(data_dir_path, d) for d in os.listdir(data_dir_path) if d not in config.dirignore]
            self.path_list += new_list
        self.N = len(self.path_list)
        self.opt = opt
        self.opt.L = self.sniff()
        self.millefeuille = Millefeuille(self.opt)
        self.tokenized = []
        print(f"listed {len(self.path_list)} data packages")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def sniff(self) -> int:
        sample_input = torch.load(os.path.join(self.path_list[0], EncodedExample.textcoded_filename))
        L = sample_input.size(2)
        return L

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int) -> Item:
        path = self.path_list[i]
        textcoded = torch.load(os.path.join(path, EncodedExample.textcoded_filename), map_location="cuda:0").float()
        features = torch.load(os.path.join(path, EncodedExample.features_filename), map_location="cuda:0").float()
        with open(os.path.join(path, EncodedExample.text_filename), 'r') as f:
            text = f.read()
        with open(os.path.join(path, EncodedExample.provenance_filename), 'r') as f:
            provenance = f.read()
        encoded_example = EncodedExample(provenance, text, features, textcoded)
        input, output, target_class = self.millefeuille.assemble(encoded_example)
        if torch.cuda.is_available():
            input = input.cuda()
            output = output.cuda()
            target_class = target_class.cuda()
        # input = input.to(self.device)
        # output = output.to(self.device)
        # target_class = target_class.to(self.device)
        return Item(text, provenance, input, output, target_class)


class Millefeuille:

    def __init__(self, opt: 'Options'):
        self.opt = opt

    def assemble(self, encoded_example: EncodedExample) -> Tuple[BxCxL, BxCxL, BxL]:
        
        # INPUT: ENCODED TEXT SAMPLES
        input = encoded_example.textcoded

        # OUTPUT SELECTION AND COMBINATION OF FEATURES
        selected_features_list = [encoded_example.features[ : , concept2index[f], : ] for f in self.opt.selected_features]
        output = torch.cat(selected_features_list, 0) # 2D C x L
        output.unsqueeze_(0) # -> 3D 1 x C x L
        # OUTPUT: add a feature for untagged characters; necessary for softmax classification
        no_tag_feature = output.sum(1) # -> 2D 1 x L, is superposition of all features so far
        no_tag_feature.unsqueeze_(0) # -> 3D 1 x 1 x L
        no_tag_feature = 1 - no_tag_feature # sets to 1 for char not tagged and to 0 for tagged characters
        output = torch.cat((output, no_tag_feature), 1) # 3D 1 x C+1 x L
        target_class = output.argmax(1) # when the output features are mutually exclusive, this allows cross_entropy or nll classification
        return input, output, target_class

def collate_fn(example_list: List[Item]) -> Minibatch:
    """
    Generates a minibatch by concatenating input and output tensors along the batch dimension. 
    This function is used in the DataLoader object.
    input and output tensors in Items should be 3D in B x C x L format.
    target_class tensor should be in B x L format.
    
    Args:
        example_list: list of Items() that form the minibatch. 
    """

    text, provenance, input, output, target_class = zip(*example_list)
    input = torch.cat(input, 0)
    output = torch.cat(output, 0)
    target_class = torch.cat(target_class, 0)
    return Minibatch(text=text, input=input, output=output, provenance=provenance, target_class=target_class)
