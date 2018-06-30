# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from smtag.loader import Dataset
import torch
from math import floor
import logging
logger = logging.getLogger(__name__)

class Minibatches: #Minibatches(Dataset)?
    '''
    Chunks a Dataset of already randomized examples into an array of minibatches.
    Minibatches is iterable and yields successively one minibatch (Dataset).
    Usage:
        minibatches = Minibatches(dataset, 128)
        for minibatch in minibatches:
            input = m.input
            target = m.output
            prediction = model(input)
            loss = loss_fn(prediction, target)
    '''
    def __init__(self, dataset, minibatch_size):
        '''
        Args:
            dataset (Dataset): the dataset to randomly split into minibatches
            minibatch_size (int): the number of examples per minibatch
        '''

        self.L = dataset.L
        self.nf_input = dataset.nf_input
        self.nf_output = dataset.nf_output
        self.minibatch_size = minibatch_size
        self.minibatch_number = floor(dataset.N / self.minibatch_size) #the rest of the examples will be ignored
        self.minibatches = []
        # test if we are on a GPU machine
        if torch.cuda.device_count() > 0: # or torch.cuda.is_available() ?
            cuda_on = True
            print("{} GPUs available!".format(torch.cuda.device_count()))
        else:
            cuda_on = False

        for i in range(self.minibatch_number):
            this_minibatch = Dataset(self.minibatch_size, self.nf_input, self.nf_output, self.L)
            #minibatch_size is 1 for online training 
            start = i * self.minibatch_size
            stop = start + self.minibatch_size
            this_minibatch.input = dataset.input[start:stop, : , : ]
            this_minibatch.output = dataset.output[start:stop, : , : ]
            this_minibatch.text = dataset.text[start:stop]
            this_minibatch.provenance = dataset.provenance[start:stop]
            #make them CUDA if necessary
            if cuda_on:
                this_minibatch.input = this_minibatch.input.cuda()
                this_minibatch.output = this_minibatch.output.cuda()
            self.minibatches.append(this_minibatch)

    #make it iterable and shuffable
    def __iter__(self):
        return self.minibatches.__iter__()

    def __next__(self):
        return next(self.minibatches)
        
    def __len__(self):
        return len(self.minibatches)
        
    def __getitem__(self, i):
        return self.minibatches[i]
        
    def __setitem__(self, i, val):
        self.minibatches[i] = val
