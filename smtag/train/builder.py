# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from math import floor
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from math import sqrt
from toolbox.models import HyperparametersCatStack, CatStack1d, Container1d
from ..common.mapper import Concept, Catalogue, concept2index
from .. import config

class HyperparemetersSmtagModel(HyperparametersCatStack):

    def __init__(self, opt=None):
        self.namebase = opt['namebase']
        self.data_path_list = opt['data_path_list']
        self.modelname = opt['modelname']
        self.learning_rate = opt['learning_rate']
        self.epochs = opt['epochs']
        self.minibatch_size = opt['minibatch_size']
        self.L = None # can only be update when loading dataset...
        self.in_channels = opt['nf_input']
        self.selected_features = Catalogue.from_list(opt['selected_features'])
        self.out_channels = len(self.selected_features)
        self.hidden_channels = opt['hidden_channels']
        self.dropout_rate = opt['dropout_rate']
        self.N_layers = opt['N_layers']
        self.kernel = opt['kernel']
        self.padding = opt['padding']
        self.stride = opt['stride']
        
        # softmax requires an <untagged> class
        self.index_of_notag_class = self.out_channels
        self.out_channels += 1

        # compatibility with Container1d
        print("out_channels=", self.out_channels)
        print("in_channels=", self.in_channels)

class SmtagModel(Container1d):

    def __init__(self, hp: HyperparemetersSmtagModel):
        # map options to attributes of standard Hyperparameter object from toolbox
        self.hp = hp
        super().__init__(self.hp, CatStack1d)
        self.output_semantics = deepcopy(self.hp.selected_features) # will be modified by adding <untagged>
        self.output_semantics.append(Catalogue.UNTAGGED) # because softmax (as included in cross_entropy loss) needs untagged class when classifying entities.
