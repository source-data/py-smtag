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
from ..datagen.context import PRETRAINED
from .. import config

class SmtagModel(nn.Module):

    def __init__(self, opt):
        super(SmtagModel, self).__init__()
        nf_input = opt.nf_input
        nf_output = opt.nf_output
        nf_table = deepcopy(opt.nf_table)
        kernel_table = deepcopy(opt.kernel_table)
        padding_table = deepcopy(opt.padding_table)
        context_table = deepcopy(opt.viz_context_table)
        context_in = PRETRAINED(torch.Tensor(1, 3, config.resized_img_size, config.resized_img_size)).numel()
        dropout = opt.dropout

        self.viz_ctxt = VizContext(context_in, context_table)
        # self.pre = nn.BatchNorm1d(nf_input)
        self.net = CatStackWithVizContext(nf_input, nf_table, kernel_table, padding_table, context_table, dropout)
        self.adapter = nn.Conv1d(self.net.out_channels, nf_output, 1, 1) # reduce output features of model to final desired number of output features
        self.BN = nn.BatchNorm1d(nf_output)
        self.output_semantics = deepcopy(opt.selected_features) # will be modified by adding <untagged>
        self.output_semantics.append(Catalogue.UNTAGGED)
        self.opt = opt

    def forward(self, x, viz_context):
        context_list = self.viz_ctxt(viz_context)
        # x = self.pre(x) # should be in embeddings
        x = self.net(x, context_list)
        x = self.adapter(x)
        x = self.BN(x)
        x = F.log_softmax(x, 1)
        # x = F.sigmoid(x)
        return x

class VizContext(nn.Module):
    def __init__(self, in_channels, context_table):
        super(VizContext, self).__init__()
        self.in_channels = in_channels
        self.context_table = context_table
        # in context_table is empty, self.linears is empty too
        self.linears = nn.ModuleList([nn.Linear(self.in_channels, out_channels) for out_channels in self.context_table])

    def forward(self, context):
        embedding_list = []
        if context.size(0) > 0: # don't compute context embedding if no context provided 
            for lin in self.linears: # skipped in self.linears empty
                ctx = lin(context) # from batch of vectors B x V to batch of embeddings B x E
                ctx = F.softmax(ctx, 1) # auto-classifies images; possibly interpretable
                ctx = ctx.unsqueeze(2) # B x E x 1
                embedding_list.append(ctx)
        return embedding_list

 
class CatStackWithVizContext(nn.Module):
    
    def __init__(self, nf_input, nf_table, kernel_table, padding_table, context_table, dropout) -> torch.Tensor:
        super(CatStackWithVizContext, self).__init__()
        self.nf_input = nf_input
        self.N_layers = len(nf_table)
        self.nf_table = nf_table
        self.kernel_table = kernel_table
        self.padding_table = padding_table
        self.context_table = context_table
        self.dropout_rate = dropout
        self.blocks = nn.ModuleList()
        self.BN_pre = nn.ModuleList()
        in_channels = self.nf_input
        cumul_channels = in_channels # features of original input included
        for i in range(self.N_layers):
            out_channels = self.nf_table[i]
            if self.context_table:
                context_channels = self.context_table[i]
            else:
                context_channels = 0
            cumul_channels += out_channels
            self.BN_pre.append(nn.BatchNorm1d(in_channels+context_channels))
            block = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Conv1d(in_channels+context_channels, out_channels, self.kernel_table[i], 1, self.padding_table[i]),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(out_channels)
            )
            self.blocks.append(block)
            in_channels = out_channels
        # self.out_channels = cumul_channels
        self.compress = nn.Conv1d(cumul_channels, 128, 1, 1)
        self.BN_out = nn.BatchNorm1d(128)
        self.out_channels = 128

    def forward(self, x: torch.Tensor, context_list) -> torch.Tensor:

        x_list = [x.clone()]
        for i in range(self.N_layers):
            if context_list: # skipped if no context_list empty in which case nf_context is also 0
                viz_context = context_list[i]
                viz_context = viz_context.repeat(1, 1, x.size(2)) # expand into B x E x L
                x = torch.cat((x, viz_context), 1) # concatenate visual context embeddings to the input B x C+E x L
                x = self.BN_pre[i](x) # or y = self.BN_pre(x); makes a difference for the final reduce(torch.cat([x,y],1))
            x = self.blocks[i](x)
            x_list.append(x.clone())
        y = torch.cat(x_list, 1)
        y = self.compress(y)
        y = self.BN_out(F.relu(y))
        return y