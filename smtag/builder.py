# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from math import floor
import torch
from torch import nn
from copy import deepcopy

class SmtagModel(nn.Module):
    
    def __init__(self, module, opt): # change this to only opt
        super(SmtagModel, self).__init__()
        self.module = module
        self.output_semantics = opt['selected_features'] 
        if 'collapsed_features' in opt:
            self.output_semantics += opt['collapsed_features'] 
        if 'overlap_features' in opt:
            self.output_semantics += opt['overlap_features']
        self.opt = opt
    
    def forward(self, x):
        return self.module.forward(x)

def build(opt):
    nf_input = opt['nf_input'] 
    nf_output = opt['nf_output']
    nf_table = deepcopy(opt['nf_table']) # need to deep copy/clone because of the pop() steps when building recursivelyl the model
    kernel_table = deepcopy(opt['kernel_table']) # need to deep copy/clone
    pool_table = deepcopy(opt['pool_table']) # need to deep copy/clone
    dropout = opt['dropout']
    pre = nn.BatchNorm1d(nf_input)
    core = nn.Sequential(Unet2(nf_input, nf_table, kernel_table, pool_table, dropout),
                            nn.Conv1d(nf_input, nf_output, 1, 1),
                            nn.BatchNorm1d(nf_output)
                        )
    post = nn.Sigmoid()
    return SmtagModel(nn.Sequential(pre, core, post), opt)

class Concat(nn.Module):
    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, tensor_sequence):
        return torch.cat(tensor_sequence, self.dim)

class Unet2(nn.Module):
    def __init__(self, nf_input, nf_table, kernel_table, pool_table, dropout):
        super(Unet2, self).__init__()
        self.level = len(kernel_table)
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
        self.dropout_rate = dropout
        self.BN_nf_input = nn.BatchNorm1d(self.nf_input, track_running_stats=True)
        self.BN_nf_output = nn.BatchNorm1d(self.nf_output, track_running_stats=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.conv_down_A = nn.Sequential(
            nn.Conv1d(self.nf_input, self.nf_input, self.kernel, 1, self.padding),
            self.BN_nf_input,
            nn.ReLU(True)
        )
        self.conv_down_B = nn.Sequential(
            nn.Conv1d(self.nf_input, self.nf_output, self.kernel, 1, self.padding),
            self.BN_nf_output,
            nn.ReLU(True)
        )
        self.conv_up_B = nn.Sequential(
            nn.ConvTranspose1d(self.nf_output, self.nf_input, self.kernel, 1, self.padding),
            self.BN_nf_input,
            nn.ReLU(True),
        )
        self.conv_up_A = nn.Sequential(
            nn.ConvTranspose1d(self.nf_input, self.nf_input, self.kernel, 1, self.padding),
            self.BN_nf_input,
            nn.ReLU(True)
        )
        if len(self.nf_table) > 0:
            self.unet2 = Unet2(self.nf_output, self.nf_table, self.kernel_table, self.pool_table, self.dropout_rate)
        else:
            self.unet2 = None
        self.concat = Concat(1)
        self.reduce = nn.Conv1d(2*self.nf_input, self.nf_input, 1, 1)
                
    def forward(self, x):
        y = self.dropout(x)
        y = self.conv_down_A(y)
        y_size_1 = y.size()
        y, pool_1_indices = nn.MaxPool1d(self.pool, self.pool, return_indices=True)(y)
        y = self.conv_down_B(y)
        
        if self.unet2 is not None:
            y_size_2 = y.size()
            y, pool_2_indices = nn.MaxPool1d(self.pool, self.pool, return_indices=True)(y)
            y = self.unet2(y)
            y = nn.MaxUnpool1d(self.pool, self.pool)(y, pool_2_indices, y_size_2)
        
        y = self.dropout(y)
        y = self.conv_up_B(y)
        y = nn.MaxUnpool1d(self.pool, self.pool)(y, pool_1_indices, y_size_1)
        y = self.conv_up_A(y)
        
        #y = x + y # residual block way simpler, less params
        #merge via concatanation of output layers followed by reducing from 2*nf_output to nf_output
        y = self.concat((x, y))
        y = self.reduce(y) 
            
        return y

