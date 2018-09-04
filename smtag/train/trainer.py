# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
from torch import nn, optim
from random import shuffle
import logging
from ..common.viz import Show, Plotter
from .evaluator import Accuracy

class Trainer:

    def __init__(self, training_minibatches, validation_minibatches, model):
        self.model = model
        # we copy the options opt and output_semantics to the trainer itself
        # in case we will need them during accuracy monitoring (for example to binarize output with feature-specific thresholds)
        # on a GPU machine, the model is wrapped into a nn.DataParallel object and the opt and output_semantics attributes would not be directly accessible
        self.opt = model.opt
        self.output_semantics = model.output_semantics #
        model_descriptor = "\n".join(["{}={}".format(k, self.opt[k]) for k in self.opt])
        print(model_descriptor)
        # wrap model into nn.DataParallel if we are on a GPU machine
        self.cuda_on = False
        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count(), "GPUs available.")
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
            self.cuda_on = True
        self.plot = Plotter() # to visualize training with some plotting device (using now TensorboardX)
        self.minibatches = training_minibatches
        self.validation_minibatches = validation_minibatches
        self.evaluator = Accuracy(self.model, self.validation_minibatches, tokenize=True)
        self.loss_fn = nn.BCELoss() # nn.SmoothL1Loss() #

    def validate(self):
        loss = 0
        for m in self.validation_minibatches: # alternatively PICK one random minibatch, probably enough
            self.model.eval()
            prediction = self.model(m.input)
            loss += self.loss_fn(prediction, m.output)
        self.model.train()
        avg_loss = loss / self.validation_minibatches.minibatch_number
        return avg_loss

    def train(self):
        self.learning_rate = self.opt['learning_rate']
        self.epochs = self.opt['epochs']
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)

        for e in range(self.epochs):
            shuffle(self.minibatches) # order of minibatches is randomized at every epoch
            avg_train_loss = 0 # loss averaged over all minibatches

            for m in self.minibatches:
                self.optimizer.zero_grad()
                prediction = self.model(m.input)
                loss = self.loss_fn(prediction, m.output)
                loss.backward()
                avg_train_loss += loss
                self.optimizer.step()

            # Logging/plotting
            avg_train_loss = avg_train_loss / self.minibatches.minibatch_number
            avg_validation_loss = self.validate() # the average loss over the validation minibatches # JUST TAKE A SAMPLE: 
            Show.example(self.validation_minibatches, self.model)
            self.plot.add_scalars("losses", {'train': avg_train_loss, 'valid': avg_validation_loss}, e) # log the losses for tensorboardX
            _, _, f1 = self.evaluator.run()
            self.plot.add_scalars("f1", {str(concept): f1[i] for i, concept in enumerate(self.output_semantics)}, e)

        #COMPUTE f1 and final loss on whole validation set
        self.plot.close()