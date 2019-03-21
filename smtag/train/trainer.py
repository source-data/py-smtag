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
from .evaluator import Accuracy
from .. import config


Minibatch = namedtuple('Minibatch', ['input', 'output', 'provenance'])

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
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "GPUs available.")
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
            self.model.output_semantics = self.output_semantics
            # self.weight = self.weight.cuda()
            self.cuda_on = True
            self.num_workers = 32 # 96
        else:
            self.cuda_on = False
            self.num_workers = 0
        self.plot = Plotter() # to visualize training with some plotting device (using now TensorboardX)
        self.batch_size = self.opt.minibatch_size
        self.trainset = trainset
        self.validation = validation
        self.trainset_minibatches = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=self.num_workers, drop_last=True)
        self.validation_minibatches = DataLoader(validation, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=self.num_workers, drop_last=True)
        self.evaluator = Accuracy(self.validation_minibatches, tokenize=False)
        self.console = Show('console')

    @staticmethod
    def collate_fn(example_list):
        provenance, input, output = zip(*example_list)
        minibatch = Minibatch(
            input = torch.cat(input, 0),
            output = torch.cat(output, 0),
            provenance = provenance
        )
        return minibatch
        

    def validate(self):
        loss = 0
        N = len(self.validation) // self.batch_size
        for i, m in enumerate(self.validation_minibatches):
            progress(i, N, "\tvalidating                              ")
            with torch.no_grad():
                self.model.eval()
                loss += self.predict(m)
                self.model.train()
        loss /= N
        return loss

    def predict(self, batch):
        x = batch.input
        y = batch.output
        if self.cuda_on:
            x = x.cuda()
            y = y.cuda()
        y_hat = self.model(x)
        loss = F.nll_loss(y_hat, y.argmax(1))
        # loss = F.binary_cross_entropy(y_hat, y)
        return loss
    
    def train(self):
        self.learning_rate = self.opt.learning_rate
        self.epochs = self.opt.epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.plot.add_text('parameters', str(self.opt))
        N = len(self.trainset) // self.batch_size
        for e in range(self.epochs):
            avg_train_loss = 0 # loss averaged over all minibatches

            for i, m in enumerate(self.trainset_minibatches):
                progress(i, N, "\ttraining epoch {}".format(e))
                self.optimizer.zero_grad()
                loss = self.predict(m)
                loss.backward()
                avg_train_loss += loss
                self.optimizer.step()

            # Logging/plotting
            print("\n")
            model_cpu = export_model(self.model, custom_name = self.opt.namebase+'_last_saved')
            avg_train_loss = avg_train_loss / N
            avg_validation_loss = self.validate() # the average loss over the validation minibatches # JUST TAKE A SAMPLE: 
            self.plot.add_scalars("losses", {'train': avg_train_loss, 'valid': avg_validation_loss}, e) # log the losses for tensorboardX
            precision, recall, f1 = self.evaluator.run(model_cpu)
            self.plot.add_scalars("f1", {str(i): f1[i] for i in range(self.opt.nf_output)}, e)
            self.plot.add_scalars("precision", {str(i): precision[i] for i in range(self.opt.nf_output)}, e)
            self.plot.add_scalars("recall", {str(i): recall[i] for i in range(self.opt.nf_output)}, e)
            self.plot.add_progress("progress", avg_train_loss, f1, self.output_semantics, e)
            print(self.console.example(self.validation_minibatches, self.model))
            # self.plot.add_example("examples", self.markdown.example(self.validation_minibatches, self.model, e)
            
        self.plot.close()
        print("\n")
        return avg_train_loss, avg_validation_loss, precision, recall, f1

