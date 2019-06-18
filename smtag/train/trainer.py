# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import sys
import resource
from collections import namedtuple
from random import randrange
from math import log
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from random import shuffle
import logging
from ..common.importexport import export_model
from ..common.viz import Show, Plotter
from ..common.progress import progress
from .. import config


Minibatch = namedtuple('Minibatch', ['input', 'output', 'viz_context', 'provenance'])

def predict_fn(model, batch, eval=False):
    x = batch.input
    y = batch.output
    viz_context = batch.viz_context
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        viz_context = viz_context.cuda()
    if eval:
        with torch.no_grad():
            model.eval()
            y_hat = model(x, viz_context)
            model.train()
    else:
        y_hat = model(x, viz_context)
    loss = F.nll_loss(y_hat, y.argmax(1))
    # loss = F.binary_cross_entropy(y_hat, y)
    return x, y, y_hat, loss

def collate_fn(example_list):
    provenance, input, output, viz_context = zip(*example_list)
    minibatch = Minibatch(
        input = torch.cat(input, 0),
        output = torch.cat(output, 0),
        viz_context = torch.cat(viz_context, 0), 
        provenance = provenance
    )
    return minibatch

from .evaluator import Accuracy # Accuracy needs predict_fn and collate_fn as well


class Trainer:

    def __init__(self, trainset, validation, model):
        self.model = model
        # we copy the options opt and output_semantics to the trainer itself
        # in case we will need them during accuracy monitoring (for example to binarize output with feature-specific thresholds)
        # on a GPU machine, the model is wrapped into a nn.DataParallel object and the opt and output_semantics attributes would not be directly accessible
        self.opt = model.opt
        self.output_semantics = model.output_semantics
        # N = len(training_minibatches)
        # B, C, L = training_minibatches[0].output.size()
        # freq = [0] * C 
        # for m in training_minibatches:
        #     for j in range(C):
        #         f = m.output[ : , j, : ]
        #         freq[j] += f.sum()
        # freq = [f/(N*B*L) for f in freq]
        # self.weight = torch.Tensor([1/f for f in freq])
        print(self.opt)
        # wrap model into nn.DataParallel if we are on a GPU machine
        self.num_workers = 0
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "GPUs available.")
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
            self.model.output_semantics = self.output_semantics
            # self.weight = self.weight.cuda()
            self.num_workers = 64
            
        self.plot = Plotter() # to visualize training with some plotting device (using now TensorboardX)
        self.batch_size = self.opt.minibatch_size
        self.trainset = trainset
        self.validation = validation
        self.trainset_minibatches = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers, drop_last=True, timeout=60)
        self.validation_minibatches = DataLoader(validation, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers, drop_last=True, timeout=60)
        self.evaluator = Accuracy(self.model, self.validation_minibatches, tokenize=False)
        self.console = Show('console')

    def train(self):
        self.learning_rate = self.opt.learning_rate
        self.epochs = self.opt.epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.plot.add_text('parameters', str(self.opt))
        N = len(self.trainset) // self.batch_size
        for e in range(self.epochs):
            avg_train_loss = 0 # loss averaged over all minibatches

            for i, batch in enumerate(self.trainset_minibatches):
                progress(i, N, "\ttraining epoch {}".format(e))
                self.optimizer.zero_grad()
                x, y, y_hat, loss = predict_fn(self.model, batch)
                loss.backward()
                avg_train_loss += loss.cpu().item() # important otherwise not freed from the graph
                self.optimizer.step()

            # Logging/plotting
            print("\n")
            export_model(self.model, custom_name = self.opt.namebase+'_last_saved')
            avg_train_loss = avg_train_loss / N
            precision, recall, f1, avg_validation_loss = self.evaluator.run(predict_fn)
            self.plot.add_scalars("losses", {'train': avg_train_loss, 'valid': avg_validation_loss}, e) # log the losses for tensorboardX
            self.plot.add_scalars("f1", {str(i): f1[i] for i in range(self.opt.nf_output)}, e)
            self.plot.add_scalars("precision", {str(i): precision[i] for i in range(self.opt.nf_output)}, e)
            self.plot.add_scalars("recall", {str(i): recall[i] for i in range(self.opt.nf_output)}, e)
            self.plot.add_progress("progress", avg_train_loss, f1, self.output_semantics, e)
            print(self.console.example(self.validation_minibatches, self.model))
            # self.plot.add_example("examples", self.markdown.example(self.validation_minibatches, self.model, e)
            
        self.plot.close()
        print("\n")
        return avg_train_loss, avg_validation_loss, precision, recall, f1

