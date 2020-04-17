# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from math import floor
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from math import sqrt
from toolbox.models import HyperparametersUnet, Unet1d, Container1d
from ..common.mapper import Concept, Catalogue, concept2index
from .. import config

class HyperparemetersSmtagModel(HyperparametersUnet):

    def __init__(self, opt=None):
        self.namebase = opt['namebase']
        self.data_path_list = opt['data_path_list']
        self.modelname = opt['modelname']
        self.learning_rate = opt['learning_rate']
        self.epochs = opt['epochs']
        self.minibatch_size = opt['minibatch_size']
        self.in_channels = opt['nf_input']
        self.selected_features = Catalogue.from_list(opt['selected_features'])
        self.out_channels = len(self.selected_features)
        self.nf_table = opt['nf_table']
        self.kernel_table = opt['kernel_table']
        self.stride_table = opt['stride_table']
        self.pool = opt['pool']
        self.dropout_rate = opt['dropout_rate']
        self.padding = opt['padding']
        
        # softmax requires an <untagged> class
        self.index_of_notag_class = self.out_channels
        self.out_channels += 1


class SmtagModel(Container1d):

    def __init__(self, hp: HyperparemetersSmtagModel):
        self.hp = hp
        super().__init__(self.hp, Unet1d)
        self.output_semantics = deepcopy(self.hp.selected_features) # will be modified by adding <untagged>
        self.output_semantics.append(Catalogue.UNTAGGED) # because softmax (as included in cross_entropy loss) needs untagged class when classifying entities.
