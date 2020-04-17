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
        self.selected_features = Catalogue.from_list(opt['selected_features'])
        super().__init__(
            in_channels = opt['nf_input'],
            out_channels = len(self.selected_features) + 1,  # softmax requires an <untagged> class
            nf_table=opt['nf_table'],
            kernel_table=opt['kernel_table'],
            stride_table=opt['stride_table'],
            dropout_rate = opt['dropout_rate'],
            pool = opt['pool']
        )
        self.index_of_notag_class = self.out_channels  # softmax requires an <untagged> class which is the last one


class SmtagModel(Container1d):

    def __init__(self, hp: HyperparemetersSmtagModel):
        self.hp = hp
        super().__init__(self.hp, Unet1d)
        self.output_semantics = deepcopy(self.hp.selected_features) # will be modified by adding <untagged>
        self.output_semantics.append(Catalogue.UNTAGGED) # because softmax (as included in cross_entropy loss) needs untagged class when classifying entities.
