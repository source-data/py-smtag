# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
from torch import nn, optim
from random import shuffle
import logging
from smtag.viz import Show, Plotter

class Trainer:

    def __init__(self, training_minibatches, validation_minibatches, model):
        self.model = model
        # we copy the options opt and output_semantics to the trainer itself
        # in case we will need them during accuracy monitoring (for example to binarize output with feature-specific thresholds)
        # on a GPU machine, the model is wrapped into a nn.DataParallel object and the opt and output_semantics attributes would not be directly accessible
        self.opt = model.opt 
        self.output_semantics = model.output_semantics # 
        model_descriptor = "\n".join(["{}={}".format(k, self.opt[k]) for k in self.model.opt])
        print(model_descriptor)
        # wrap model into nn.DataParallel if we are on a GPU machine
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count(), "GPUs available.")
            self.model = nn.DataParallel(self.model)
        self.model.to(device)
        self.plot = Plotter() # to visualize training with some plotting device (using now TensorboardX)
        self.minibatches = training_minibatches
        self.validation_minibatches = validation_minibatches
        self.loss_fn = nn.BCELoss() # nn.SmoothL1Loss() # 

    def validate(self):
        self.model.eval()
        avg_loss = 0

        for m in self.validation_minibatches:
            input, target = m.input, m.output
            prediction = self.model(input)
            loss = self.loss_fn(prediction, target)
            avg_loss += loss

        self.model.train()
        avg_loss = avg_loss / len(self.validation_minibatches)

        return avg_loss

    def train(self):
        opt = self.model.opt
        self.learning_rate = opt['learning_rate']
        self.epochs = opt['epochs']
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        
        for e in range(self.epochs):
            shuffle(self.minibatches) # order of minibatches is randomized at every epoch
            avg_train_loss = 0 # loss averaged over all minibatches

            counter = 1
            for m in self.minibatches:
                input, target = m.input, m.output
                self.optimizer.zero_grad()
                prediction = self.model(input)
                loss = self.loss_fn(prediction, target)
                loss.backward()
                avg_train_loss += loss
                self.optimizer.step()
                print("\n\n\nepoch {}\tminibatch #{}\tloss={}".format(e, counter, loss))
                Show.example(self.validation_minibatches, self.model)
                counter += 1

            # Logging/plotting
            avg_train_loss = avg_train_loss / self.minibatches.minibatch_number
            avg_validation_loss = self.validate() # the average loss over the validation minibatches
            self.plot.add_losses({'train':avg_train_loss, 'valid':avg_validation_loss}, e) # log the losses for tensorboardX
            #Log values and gradients of the parameters (histogram summary)
            #for name, param in self.model.named_parameters():
            #    name = name.replace('.', '/')
            #    self.writer.add_histogram(name, param.clone().cpu().data.numpy(), e)
            #    self.writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), e)
        self.plot.close()
