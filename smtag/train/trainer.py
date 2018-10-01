# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import sys
import resource
import gc
import torch
from torch import nn, optim
from random import shuffle
import logging
from ..common.importexport import export_model
from ..common.viz import Show, Plotter
from ..common.progress import progress
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
        if torch.cuda.device_count() > 0:
            print(torch.cuda.device_count(), "GPUs available.")
            self.model = nn.DataParallel(self.model,)
            self.model.cuda()
            self.model.output_semantics = self.output_semantics
            self.cuda_on = True
        else:
            self.cuda_on = False
        self.plot = Plotter() # to visualize training with some plotting device (using now TensorboardX)
        self.minibatches = training_minibatches
        self.validation_minibatches = validation_minibatches
        self.evaluator = Accuracy(self.model, self.validation_minibatches, tokenize=False)
        self.loss_fn = nn.BCELoss()
        self.markdown = Show('markdown')
        self.console = Show('console')

    def validate(self):
        loss = 0
        for m in self.validation_minibatches: # alternatively PICK one random minibatch, probably enough
            m_input = m.input
            m_output = m.output
            if self.cuda_on:
                m_input = m_input.cuda()
                m_output = m_output.cuda()
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(m_input)
                loss += self.loss_fn(prediction, m_output)
        self.model.train()
        avg_loss = loss / self.validation_minibatches.minibatch_number
        return avg_loss

    def train(self):
        self.learning_rate = self.opt['learning_rate']
        self.epochs = self.opt['epochs']
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.plot.add_text('parameters', "; ".join([o+"="+str(self.opt[o]) for o in self.opt]))
        N = self.minibatches.minibatch_number
        for e in range(self.epochs):
            shuffle(self.minibatches) # order of minibatches is randomized at every epoch
            avg_train_loss = 0 # loss averaged over all minibatches

            for i, m in enumerate(self.minibatches):
                progress(i, N, "\ttraining epoch {}".format(e))
                m_input = m.input
                m_output = m.output
                if self.cuda_on:
                    m_input = m_input.cuda()
                    m_output = m_output.cuda()
                self.optimizer.zero_grad()
                prediction = self.model(m_input)
                loss = self.loss_fn(prediction, m_output)
                loss.backward()
                avg_train_loss += loss
                self.optimizer.step()

            # Logging/plotting
            avg_train_loss = avg_train_loss / N
            avg_validation_loss = self.validate() # the average loss over the validation minibatches # JUST TAKE A SAMPLE: 
            self.plot.add_scalars("losses", {'train': avg_train_loss, 'valid': avg_validation_loss}, e) # log the losses for tensorboardX
            precision, recall, f1 = self.evaluator.run()
            self.plot.add_scalars("f1", {str(concept): f1[i] for i, concept in enumerate(self.output_semantics)}, e)
            self.plot.add_progress("progress", avg_train_loss, f1, self.output_semantics, e)
            print(self.console.example(self.validation_minibatches, self.model))
            # self.plot.add_example("examples", self.markdown.example(self.validation_minibatches, self.model, e)
            export_model(self.model, custom_name = self.opt['namebase']+'_last_saved')
        self.plot.close()
        print("\n")
        return avg_train_loss, avg_validation_loss, precision, recall, f1

