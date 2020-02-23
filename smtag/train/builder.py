# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from math import floor
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from math import sqrt
from toolbox.models import Container1d, CatStack1d, HyperparametersCatStack
from ..common.options import Options
from ..common.mapper import Concept, Catalogue
from .. import config

class SmtagModel(Container1d):

    def __init__(self, opt: Options):
        super().__init__(opt.hp)
        self.output_semantics = deepcopy(opt.selected_features) # will be modified by adding <untagged>
        self.output_semantics.append(Catalogue.UNTAGGED) # because softmax (as included in cross_entropy loss) needs untagged class when classifying entities.
        self.opt = opt

