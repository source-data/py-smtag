# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from math import floor
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from smtag.mapper import Catalogue

class SmtagModel(nn.Module):

    def __init__(self, opt):
        super(SmtagModel, self).__init__()
        nf_input = opt['nf_input'] 
        nf_output = opt['nf_output']
        nf_table = deepcopy(opt['nf_table']) # need to deep copy/clone because of the pop() steps when building recursivelyl the model
        kernel_table = deepcopy(opt['kernel_table']) # need to deep copy/clone
        pool_table = deepcopy(opt['pool_table']) # need to deep copy/clone
        dropout = opt['dropout']
        
        self.pre = nn.BatchNorm1d(nf_input)
        self.unet = Unet2(nf_input, nf_table, kernel_table, pool_table, dropout)
        self.adapter = nn.Conv1d(nf_input, nf_output, 1, 1)
        self.BN = nn.BatchNorm1d(nf_output)
        
        self.output_semantics = Catalogue.from_list(opt['selected_features']) 
        if 'collapsed_features' in opt:
            #print(opt['collapsed_features'])
            if opt['collapsed_features']:
                # WARNING! keep only the first one by convention. NO GREAT. LACK OF STRUCTURE IN FEATURE SEMANTICS. CATEGORY, TYPE, ROLE
                self.output_semantics.append(Catalogue.from_label(opt['collapsed_features'][0]))
        if 'overlap_features' in opt:
             #print(opt['overlap_features'])
             if opt['overlap_features']:
                 # WARNING! keep only the first one by convention. NO GREAT. LACK OF STRUCTURE IN FEATURE SEMANTICS. CATEGORY, TYPE, ROLE
                 # for example if model trained with meta -a geneprod,reporter, the resulting model will carry only GENEPROD as its output semantics
                 self.output_semantics.append(Catalogue.from_label(opt['overlap_features'][0]))
        self.opt = opt

    def forward(self, x):
        # print("in forward", " x ".join([str(n) for n in list(x.size())]))
        x = self.pre(x)
        x = self.unet(x)
        x = self.adapter(x)
        x = self.BN(x)
        x = F.sigmoid(x)
        return x

class Concat(nn.Module):
    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, tensor_sequence):
        return torch.cat(tensor_sequence, self.dim)

class Unet2(nn.Module):
    def __init__(self, nf_input, nf_table, kernel_table, pool_table, dropout_rate):
        super(Unet2, self).__init__()
        self.nf_input = nf_input
        self.nf_table = nf_table
        self.nf_output = nf_table.pop(0)
        self.pool_table = pool_table
        self.pool = pool_table.pop(0)
        self.kernel_table = kernel_table
        self.kernel = kernel_table.pop(0)
        if self.kernel % 2 == 0:
           self.padding = int(self.kernel/2)
        else:
           self.padding = floor((self.kernel-1)/2)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.conv_down_A = nn.Conv1d(self.nf_input, self.nf_input, self.kernel, 1, self.padding)
        self.BN_down_A = nn.BatchNorm1d(self.nf_input)

        self.conv_down_B = nn.Conv1d(self.nf_input, self.nf_output, self.kernel, 1, self.padding)
        self.BN_down_B = nn.BatchNorm1d(self.nf_output)

        self.conv_up_B = nn.ConvTranspose1d(self.nf_output, self.nf_input, self.kernel, 1, self.padding)
        self.BN_up_B = nn.BatchNorm1d(self.nf_input)

        self.conv_up_A = nn.ConvTranspose1d(self.nf_input, self.nf_input, self.kernel, 1, self.padding)
        self.BN_up_A = nn.BatchNorm1d(self.nf_input)

        if len(self.nf_table) > 0:
            self.unet2 = Unet2(self.nf_output, self.nf_table, self.kernel_table, self.pool_table, self.dropout_rate)
        else:
            self.unet2 = None
        self.concat = Concat(1)
        self.reduce = nn.Conv1d(2*self.nf_input, self.nf_input, 1, 1)

    def forward(self, x):
        
        y = self.dropout(x)
        y = self.conv_down_A(y)
        y = F.relu(self.BN_down_A(y))
        y_size_1 = y.size()
        y, pool_1_indices = nn.MaxPool1d(self.pool, self.pool, return_indices=True)(y)
        y = self.conv_down_B(y)
        y = F.relu(self.BN_down_B(y))

        if self.unet2 is not None:
            y_size_2 = y.size()
            y, pool_2_indices = nn.MaxPool1d(self.pool, self.pool, return_indices=True)(y)
            y = self.unet2(y)
            y = nn.MaxUnpool1d(self.pool, self.pool)(y, pool_2_indices, y_size_2)

        y = self.dropout(y)
        y = self.conv_up_B(y)
        y = F.relu(self.BN_up_B(y))
        y = nn.MaxUnpool1d(self.pool, self.pool)(y, pool_1_indices, y_size_1)
        y = self.conv_up_A(y)
        y = F.relu(self.BN_up_A(y))

        #y = x + y # this is the residual block way of making the shortcut through the branche of the U; simpler, less params, no need for self.reduce()
        y = self.concat((x, y)) # merge via concatanation of output layers followed by reducing from 2*nf_output to nf_output
        y = self.reduce(y) 

        return y

