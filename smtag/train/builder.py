# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from math import floor
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from math import sqrt
from toolbox.models import Container1d, CatStack1d, HyperparametersCatStack, Hyperparameters
from ..common.options import Options
from ..common.mapper import Concept, Catalogue
from .. import config

class SmtagModel(Container1d):

    def __init__(self, opt: Options):
        # map options to attributes of standard Hyperparameter object from toolbox
        hp = HyperparametersCatStack(
            in_channels = opt.nf_input,
            hidden_channels = opt.hidden_channels,
            out_channels = opt.nf_output,
            dropout_rate = opt.dropout_rate,
            N_layers = opt.N_layers,
            kernel = opt.kernel,
            padding = opt.padding,
            stride = opt.stride,
        )
        super().__init__(hp, CatStack1d)
        self.output_semantics = deepcopy(opt.selected_features) # will be modified by adding <untagged>
        self.output_semantics.append(Catalogue.UNTAGGED) # because softmax (as included in cross_entropy loss) needs untagged class when classifying entities.
        self.opt = opt
