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
from ..common.importexport import load_smtag_model, export_smtag_model
from ..train.builder import SmtagModel
from ..train.dataset import collate_fn, BxCxL, BxL, Minibatch, Data4th
from ..predict.predictor import predict_fn
from ..common.progress import progress
from .. import config
from .evaluator import Accuracy
from ..common.viz import Show

class Trainer:

    def __init__(self, trainset: Data4th, validation: Data4th, model: SmtagModel):
        self.model = model
        # assigning model.opt to self.opt because on a GPU machine, the model is wrapped into a nn.DataParallel object and its opt attribute would not be directly accessible
        self.hp = self.model.hp
        self.namebase = "_".join([str(f) for f in self.hp.selected_features]) # used to save models to disk
        print(self.model.hp)
        # wrap model into nn.DataParallel if we are on a GPU machine
        self.num_workers = 0
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "GPUs available for training.")
            gpu_model = self.model.cuda()
            gpu_model = nn.DataParallel(gpu_model)
            self.model = gpu_model
            self.num_workers = os.cpu_count()
        self.batch_size = self.hp.minibatch_size
        self.trainset = trainset
        self.validation = validation
        self.trainset_minibatches = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers, drop_last=True, timeout=60)
        self.validation_minibatches = DataLoader(validation, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers, drop_last=True, timeout=60)
        self.evaluator = Accuracy(self.model, self.validation_minibatches, self.hp.out_channels)
        self.plot = SummaryWriter() # to visualize training
        self.console = Show('console') # to output training progress to the console

    def train(self) -> Tuple[SmtagModel, float]:
        self.learning_rate = self.hp.learning_rate
        self.epochs = self.hp.epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.plot.add_text('parameters', str(self.hp))
        N = len(self.trainset_minibatches) # the number of minibatches
        for e in range(self.epochs):
            avg_train_loss = 0 # loss averaged over all minibatches
            for i, batch in enumerate(self.trainset_minibatches):
                progress(i, N, "\ttraining epoch {}".format(e))
                self.optimizer.zero_grad()
                y, y_hat, loss = predict_fn(self.model, batch, eval=False)
                loss.backward()
                avg_train_loss += loss.cpu().item() # important otherwise not freed from the graph
                self.optimizer.step()

            # Logging/plotting
            print("\n")
            avg_train_loss = avg_train_loss / N
            precision, recall, f1, avg_validation_loss = self.evaluator.run(predict_fn)
            self.plot.add_scalars("data/losses", {'train': avg_train_loss, 'valid': avg_validation_loss}, e) # log the losses for tensorboardX
            self.plot.add_scalars("data/f1", {str(i): f1[i] for i in range(self.hp.out_channels)}, e)
            self.plot.add_scalars("data/precision", {str(i): precision[i] for i in range(self.hp.out_channels)}, e)
            self.plot.add_scalars("data/recall", {str(i): recall[i] for i in range(self.hp.out_channels)}, e)
            self.console.example(self.validation, self.model)
            model_name = self.namebase + f"_epoch_{e:03d}"
            model_path = export_smtag_model(self.model, model_name)
        self.plot.close()
        print("\n")
        return self.model, model_path, precision, recall, f1, avg_validation_loss
