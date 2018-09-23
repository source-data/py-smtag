# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from math import pow
from random import randint
from ..common.importexport import export_model

class HyperScan():

    def __init__(self, opt, path):
        self._metrics = []
        self.opt = opt
        self.path = path # path to save scan results

    def append(self, model, perf, id):
        self._metrics.append(perf)
        filename = 'scanned_model_' + str(id)
        export_model(model, filename, model_dir = self.path)

    def randopt(self, selected_hyperparam = {'log_lr'}):
        """
        Model and training hyperparameter are sampled randomly given the specified ranges.

        Args: 
            selected_hyperparamters (str) from 'log_lr' (-3) | 'log_batch_size' (5) | 'depth' (3) | 'nf' (8) | 'kernel' (6) | 'pooling' (2)

        Returns:
            dict where selected hyperparameters where randomly sampled
        """

        # default hyperparam
        hparam = {
            'log_lr': -3,
            'log_batch_size': 5,
            'depth': 3,
            'nf': 8,
            'kernel': 6,
            'pooling': 2
        }
        #randomly sampling hyperparameters
        randparam = {
            'log_lr': randint(-4, -1),
            'log_batch_size': randint(4, 8),
            'depth': randint(2,4),
            'nf': randint(4,16),
            'kernel': randint(3,10),
            'pooling': randint(1,3),
        }

        for h in selected_hyperparam: 
            hparam[h] = randparam[h]


        self.opt['learning_rate'] = 10 ** hparam['log_lr']
        self.opt['minibatch_size'] =  2 ** hparam['log_batch_size']
        nf_table = []
        kernel_table = []
        pool_table =[]
        for _ in range(hparam['depth']):
            nf_table.append(hparam['nf'])
            kernel_table.append(hparam['kernel'])
            pool_table.append(hparam['pooling'])
        self.opt['nf_table'] = nf_table
        self.opt['kernel_table'] = kernel_table
        self.opt['pool_table'] = pool_table
        return self.opt


    def threshold(self):
        """
        scanning thresholds for binarization of output
        """
        pass
        #WIP
        # #model_basename
        #benchmark = Accuracy(self.model, testset, tokenize=self.tokenize)
        #self.precision, self.recall, self.f1 = benchmark.run()