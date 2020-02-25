# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import sys
import os
import resource
from collections import namedtuple
from random import randrange
from math import log
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from random import shuffle
from typing import Tuple
from tensorboardX import SummaryWriter
from ..common.importexport import export_model
from ..train.builder import SmtagModel
from ..train.dataset import collate_fn, BxCxL, BxL, Minibatch, Data4th
from ..common.progress import progress
from .. import config


def predict_fn(model: SmtagModel, batch: Minibatch, eval: bool=False) -> Tuple[BxCxL, BxL, BxCxL, torch.Tensor]:
    """
    Prediction function used during training or evaluation of a model. 

    Artgs:
        model (SmtagModel): the model to be used for the prediction.
        batch (Minibatch): a minibatch of examples with input, output, target_class and provenance.
        eval (bool): flag to specify if the model is used in training or evaluation mode.
    
    Returns:
        input tensor (BxCxL), target class tensor (BxL), predicted tensor (BxCxL), loss (torch.Tensor)
    """
    x = batch.input
    y = batch.target_class
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    if eval:
        with torch.no_grad():
            model.eval()
            y_hat = model(x)
            model.train()
    else:
        y_hat = model(x)
    loss = F.cross_entropy(y_hat, y) # y is a target class tensor BxL
    return y_hat, loss

from .evaluator import Accuracy # Imported only now because Accuracy needs predict_fn().
from ..common.viz import Show

class Trainer:

    def __init__(self, trainset: Data4th, validation: Data4th, model: SmtagModel):
        self.model = model
        # assigning model.opt to self.opt because on a GPU machine, the model is wrapped into a nn.DataParallel object and its opt attribute would not be directly accessible
        self.opt = self.model.opt
        self.namebase = "_".join([str(f) for f in self.opt.selected_features]) # used to save models to disk
        print(self.model.opt)
        # wrap model into nn.DataParallel if we are on a GPU machine
        self.num_workers = 0
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "GPUs available.")
            gpu_model = self.model.cuda()
            gpu_model = nn.DataParallel(gpu_model)
            self.model = gpu_model
            self.num_workers = os.cpu_count()
        self.batch_size = self.opt.minibatch_size
        self.trainset = trainset
        self.validation = validation
        self.trainset_minibatches = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers, drop_last=True, timeout=60)
        self.validation_minibatches = DataLoader(validation, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers, drop_last=True, timeout=60)
        self.evaluator = Accuracy(self.model, self.validation_minibatches, self.opt.nf_output)
        self.plot = SummaryWriter() # to visualize training
        self.console = Show('console') # to output training progress to the console

    def train(self) -> Tuple[SmtagModel, float]:
        self.learning_rate = self.opt.learning_rate
        self.epochs = self.opt.epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.plot.add_text('parameters', str(self.opt))
        N = len(self.trainset_minibatches) # the number of minibatches
        for e in range(self.epochs):
            avg_train_loss = 0 # loss averaged over all minibatches
            for i, batch in enumerate(self.trainset_minibatches):
                progress(i, N, "\ttraining epoch {}".format(e))
                self.optimizer.zero_grad()
                y_hat, loss = predict_fn(self.model, batch)
                loss.backward()
                avg_train_loss += loss.cpu().item() # important otherwise not freed from the graph
                self.optimizer.step()

            # Logging/plotting
            print("\n")
            avg_train_loss = avg_train_loss / N
            precision, recall, f1, avg_validation_loss = self.evaluator.run(predict_fn)
            self.plot.add_scalars("data/losses", {'train': avg_train_loss, 'valid': avg_validation_loss}, e) # log the losses for tensorboardX
            self.plot.add_scalars("data/f1", {str(i): f1[i] for i in range(self.opt.nf_output)}, e)
            self.plot.add_scalars("data/precision", {str(i): precision[i] for i in range(self.opt.nf_output)}, e)
            self.plot.add_scalars("data/recall", {str(i): recall[i] for i in range(self.opt.nf_output)}, e)
            self.console.example(self.validation, self.model)
            export_model(self.model, self.namebase + f"_epoch_{e:03d}")
        self.plot.close()
        print("\n")
        return self.model, precision, recall, f1, avg_validation_loss
