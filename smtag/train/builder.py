# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from math import floor
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from math import sqrt
from ..common.mapper import Concept, Catalogue
from .. import config

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
        nf_input = opt.nf_input
        nf_output = opt.nf_output
        nf_table = deepcopy(opt.nf_table) # need to deep copy/clone because of the pop() steps when building recursivelyl the model
        kernel_table = deepcopy(opt.kernel_table) # need to deep copy/clone
        pool_table = deepcopy(opt.pool_table) # need to deep copy/clone
        context_table = deepcopy(opt.viz_context_table)  # need to deep copy/clone
        context_in = opt.viz_context_features # from ..datagen.context import PRETRAINED; PRETRAINED(torch.Tensor([1, config.resized_img_size, config.resized_img_size])).numelement()
        dropout = opt.dropout
        skip = opt.skip

        self.viz_ctxt = Context(context_in, context_table)
        self.pre = nn.BatchNorm1d(nf_input, track_running_stats=BNTRACK, affine=AFFINE)
        self.unet = Unet2(nf_input, nf_table, kernel_table, pool_table, context_table, dropout, skip)
        self.adapter = nn.Conv1d(nf_input, nf_output, 1, 1, bias=BIAS) # reduce output features of unet to final desired number of output features
        self.BN = nn.BatchNorm1d(nf_output, track_running_stats=BNTRACK, affine=AFFINE)
        self.output_semantics = deepcopy(opt.selected_features) # will be modified by adding <untagged>
        self.output_semantics.append(Catalogue.UNTAGGED)
        self.opt = opt

    def forward(self, x, viz_context):
        context_list = self.viz_ctxt(viz_context)
        # x = self.pre(x) # not sure about this
        x = self.unet(x, context_list)
        x = self.adapter(x)
        x = self.BN(x)
        x = F.log_softmax(x, 1)
        # x = F.sigmoid(x)
        return x

class Concat(nn.Module):
    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, tensor_sequence):
        return torch.cat(tensor_sequence, self.dim)

class Context(nn.Module):
    def __init__(self, in_channels, context_table):
        super(Context, self).__init__()
        self.in_channels = in_channels
        self.context_table = context_table
        self.linears = nn.ModuleList([nn.Linear(self.in_channels, out_channels) for out_channels in self.context_table])

    def forward(self, context):
        embedding_list = []
        if context.size(0) > 0:
            for lin in self.linears:
                ctx = lin(context) # from batch of vectors B x V to batch of embeddings B x E
                ctx = F.softmax(ctx, 1) # auto-classifies images; possibly interpretable
                ctx = ctx.unsqueeze(2) # B x E x 1
                embedding_list.append(ctx)
        return embedding_list

class Unet2(nn.Module):
    def __init__(self, nf_input, nf_table, kernel_table, pool_table, context_table, dropout_rate, skip=True):
        super(Unet2, self).__init__()
        self.context_table = context_table
        self.nf_input = nf_input
        self.nf_context = 0
        if self.context_table:
            self.nf_context = context_table.pop(0)
        self.nf_table = nf_table
        self.nf_output = nf_table.pop(0)
        self.pool_table = pool_table
        self.pool = pool_table.pop(0)
        self.kernel_table = kernel_table
        self.kernel = kernel_table.pop(0)
        # if self.kernel % 2 == 0:
        #    self.padding = int(self.kernel/2)
        # else:
        #    self.padding = floor((self.kernel-1)/2) # TRY WITHOUT ANY PADDING
        self.padding = 0 
        self.stride = 1
        self.dropout_rate = dropout_rate
        self.skip = skip
        self.dropout = nn.Dropout(self.dropout_rate)
        self.BN_context = nn.BatchNorm1d(self.nf_input+self.nf_context, track_running_stats=BNTRACK, affine=AFFINE)
        self.conv_down_A = nn.Conv1d(self.nf_input+self.nf_context, self.nf_input+self.nf_context, self.kernel, self.stride, self.padding, bias=BIAS)
        self.BN_down_A = nn.BatchNorm1d(self.nf_input+self.nf_context, track_running_stats=BNTRACK, affine=AFFINE)

        self.conv_down_B = nn.Conv1d(self.nf_input+self.nf_context, self.nf_output, self.kernel, self.stride, self.padding, bias=BIAS)
        self.BN_down_B = nn.BatchNorm1d(self.nf_output, track_running_stats=BNTRACK, affine=AFFINE)

        self.conv_up_B = nn.ConvTranspose1d(self.nf_output, self.nf_input+self.nf_context, self.kernel, self.stride, self.padding, bias=BIAS)
        self.BN_up_B = nn.BatchNorm1d(self.nf_input+self.nf_context, track_running_stats=BNTRACK, affine=AFFINE)

        self.conv_up_A = nn.ConvTranspose1d(self.nf_input+self.nf_context, self.nf_input+self.nf_context, self.kernel, self.stride, self.padding, bias=BIAS)
        self.BN_up_A = nn.BatchNorm1d(self.nf_input+self.nf_context, track_running_stats=BNTRACK, affine=AFFINE)

        if len(self.nf_table) > 0:
            self.unet2 = Unet2(self.nf_output, self.nf_table, self.kernel_table, self.pool_table, context_table, self.dropout_rate, self.skip)
            self.BN_middle = nn.BatchNorm1d(self.nf_output, track_running_stats=BNTRACK, affine=AFFINE)
        else:
            self.unet2 = None

        if self.skip:
            self.concat = Concat(1)
            self.reduce = nn.Conv1d(2*(self.nf_input+self.nf_context), self.nf_input, 1, 1)

    def forward(self, x, context_list):
        if context_list:
            viz_context = context_list[0]
            viz_context = viz_context.repeat(1, 1, x.size(2)) # expand into B x E x L
            x = torch.cat((x, viz_context), 1) # concatenate visual context embeddings to the input B x C+E x L
            # need to normalize this together? output of densenet161 is normalized but scale of x can be very different if internal layer of U-net
            x = self.BN_context(x)
            context_list = context_list[1:]
        y = self.dropout(x)
        y = self.conv_down_A(y)
        y = F.relu(self.BN_down_A(y), inplace=True)
        y_size_1 = list(y.size())
        y, pool_1_indices = nn.MaxPool1d(self.pool, self.pool, return_indices=True)(y)
        y = self.conv_down_B(y)
        y = F.relu(self.BN_down_B(y), inplace=True)

        if self.unet2 is not None:
            y_size_2 = list(y.size())
            y, pool_2_indices = nn.MaxPool1d(self.pool, self.pool, return_indices=True)(y)
            y = self.unet2(y, context_list)
            y = F.relu(self.BN_middle(y), inplace=True)
            y = nn.MaxUnpool1d(self.pool, self.pool)(y, pool_2_indices, y_size_2) # problem on 1.0.1 ses issue #16486
        y = self.dropout(y)
        y = self.conv_up_B(y)
        y = F.relu(self.BN_up_B(y), inplace=True)
        y = nn.MaxUnpool1d(self.pool, self.pool)(y, pool_1_indices, y_size_1)
        y = self.conv_up_A(y)
        y = F.relu(self.BN_up_A(y), inplace=True)

        if self.skip:
            y = self.concat((x, y)) # merge via concatanation of output layers 
            y = self.reduce(y) # reducing from 2*nf_output to nf_output
            # y = x + y # this would be the residual block way of making the shortcut through the branche of the U; simpler, less params, no need for self.reduce()

        return y
