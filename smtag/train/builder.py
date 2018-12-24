# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from math import floor
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from math import sqrt
from ..common.mapper import Concept, Catalogue

BNTRACK = True
AFFINE = True
BIAS =  True

# class BNL(nn.Module):
#     def __init__(self, channels, p=1, affine=AFFINE):
#         super(BNL, self).__init__()
#         self.channels = channels
#         self.eps = torch.Tensor(1, channels, 1).fill_(1E-05)
#         if affine:
#             self.gamma = nn.Parameter(torch.Tensor(1, channels, 1).uniform_(0,1))
#             self.beta = nn.Parameter(torch.zeros(1, channels, 1))
#         else:
#             self.gamma = 1.
#             self.beta = 0.
#         self.p = p

#     def forward(self, x):
#         mu = x.mean(2, keepdim=True).mean(0, keepdim=True)
#         y = x - mu
#         L_p = y.norm(p=self.p, dim=2, keepdim=True).mean(0, keepdim=True)
#         # L_p = y.std(2, keepdim = True).mean(0, keepdim=True)
#         y = y / (L_p + self.eps)
#         y = self.gamma * x + self.beta
#         return y

class SmtagModel(nn.Module):

    def __init__(self, opt):
        super(SmtagModel, self).__init__()
        nf_input = opt['nf_input']
        nf_output = opt['nf_output']
        nf_table = deepcopy(opt['nf_table']) # need to deep copy/clone because of the pop() steps when building recursivelyl the model
        kernel_table = deepcopy(opt['kernel_table']) # need to deep copy/clone
        pool_table = deepcopy(opt['pool_table']) # need to deep copy/clone
        dropout = opt['dropout']

        self.pre = nn.BatchNorm1d(nf_input, track_running_stats=BNTRACK, affine=AFFINE)
        self.unet = Unet2(nf_input, nf_table, kernel_table, pool_table, dropout)
        self.adapter = nn.Conv1d(nf_input, nf_output, 1, 1, bias=BIAS) # reduce output features of unet to final desired number of output features
        self.BN = nn.BatchNorm1d(nf_output, track_running_stats=BNTRACK, affine=AFFINE)

        self.output_semantics = Catalogue.from_list(opt['selected_features'])
        if 'collapsed_features' in opt:
            if opt['collapsed_features']:
                concepts = [Catalogue.from_label(f) for f in opt['collapsed_features']]
                collapsed_concepts = Concept()
                for c in concepts:
                    collapsed_concepts += c # __add__ operation defined in mapper, complements or concatenates types, roles and serialization recipes; maybe misleading because not commutative?
                self.output_semantics.append(collapsed_concepts)
        if 'overlap_features' in opt:
             if opt['overlap_features']:
                concepts = [Catalogue.from_label(f) for f in opt['overlap_features']]
                overlap_features = Concept()
                for c in concepts:
                    overlap_features += c # __add__ operation defined in mapper, complements or concatenates types, roles and serialization recipes; maybe misleading because not commutative?
                self.output_semantics.append(overlap_features)
        self.opt = opt

    def forward(self, x):
        # x = self.pre(x)
        x = self.unet(x)
        x = self.adapter(x)
        x = self.BN(x)
        x = torch.sigmoid(x)
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
           self.padding = floor((self.kernel-1)/2) # TRY WITHOUT ANY PADDING
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        stride = 1
        self.conv_down_A = nn.Conv1d(self.nf_input, self.nf_input, self.kernel, stride, self.padding, bias=BIAS)
        self.BN_down_A = nn.BatchNorm1d(self.nf_input, track_running_stats=BNTRACK, affine=AFFINE)

        self.conv_down_B = nn.Conv1d(self.nf_input, self.nf_output, self.kernel, stride, self.padding, bias=BIAS)
        self.BN_down_B = nn.BatchNorm1d(self.nf_output, track_running_stats=BNTRACK, affine=AFFINE)

        self.conv_up_B = nn.ConvTranspose1d(self.nf_output, self.nf_input, self.kernel, stride, self.padding, bias=BIAS)
        self.BN_up_B = nn.BatchNorm1d(self.nf_input, track_running_stats=BNTRACK, affine=AFFINE)

        self.conv_up_A = nn.ConvTranspose1d(self.nf_input, self.nf_input, self.kernel, stride, self.padding, bias=BIAS)
        self.BN_up_A = nn.BatchNorm1d(self.nf_input, track_running_stats=BNTRACK, affine=AFFINE)

        if len(self.nf_table) > 0:
            self.unet2 = Unet2(self.nf_output, self.nf_table, self.kernel_table, self.pool_table, self.dropout_rate)
            self.BN_middle = nn.BatchNorm1d(self.nf_output, track_running_stats=BNTRACK, affine=AFFINE)
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
            y = F.relu(self.BN_middle(y))
            y = nn.MaxUnpool1d(self.pool, self.pool)(y, pool_2_indices, y_size_2)
        y = self.dropout(y)
        y = self.conv_up_B(y)
        y = F.relu(self.BN_up_B(y))
        y = nn.MaxUnpool1d(self.pool, self.pool)(y, pool_1_indices, y_size_1)
        y = self.conv_up_A(y)
        y = F.relu(self.BN_up_A(y))

        # y = x + y # this is the residual block way of making the shortcut through the branche of the U; simpler, less params, no need for self.reduce()
        y = self.concat((x, y)) # merge via concatanation of output layers 
        y = self.reduce(y) # reducing from 2*nf_output to nf_output

        return y
