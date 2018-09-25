# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import os
from random import randint, uniform
from datetime import datetime
from ..common.importexport import export_model
from ..common.utils import cd

NL = "\n"
SEP= "\t"

class HyperScan():

    def __init__(self, opt, dir_name):
        """
        scans
            todays_scan
                models
                    scanned_model_1.zip
                    scanned_model_2.zip
                    ...
                scanned_perf.csv
            yesterds_scan
                models
                    ...
                scanned_perf.scv
        """

        self._metrics = []
        self.opt = opt
        timestamp = datetime.now().isoformat("-",timespec='minutes').replace(":", "-") # dir to save scan results
        self.dir_name = dir_name + "_" + timestamp 
        if not os.path.isdir(self.dir_name):
            os.mkdir(self.dir_name)
        self.perf_path = os.path.join(self.dir_name, 'scanned_perf.csv')
        with cd(self.dir_name):
            if not os.path.isdir('models'):
                os.mkdir('models')
        self.scanned_models_path = os.path.join(self.dir_name, 'models')


    def append(self, model, perf, opt, id):
        perf['precision'] = perf['precision'].mean()
        perf['recall'] = perf['recall'].mean()
        perf['f1'] = perf['f1'].mean()
        perf['train_loss'] = perf['train_loss'].mean()
        perf['valid_loss'] = perf['valid_loss'].mean()
        self._metrics.append(perf)
        model_filename = 'scanned_model_' + str(id)
        export_model(model, model_filename, model_dir = self.scanned_models_path)
        self.save(perf, opt, self.perf_path)

    @staticmethod
    def save(row, opt, mypath):
        if os.path.isfile(mypath):
            mode = 'a' # append row at the end of the file
        else:
            mode = 'w' # create and write to file for the first time
        with open(mypath, mode) as f:
            if mode == 'w': 
                # write a header line when writing for the first time to the file
                header_params = SEP.join([k for k in opt])
                header_results = SEP.join([k for k in row])
                header = SEP.join([header_params, header_results])
                f.write(header+NL)
                print(header)
            line_params = SEP.join([str(opt[k]) for k in opt])
            line_results = SEP.join(["{:.3f}".format(row[k]) for k in row])
            line = SEP.join([line_params, line_results])
            f.write(line+NL)
            print(line)


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
            'log_lr': uniform(-4, -1),
            'log_batch_size': uniform(4, 8),
            'depth': randint(2,4),
            'nf': randint(4,16),
            'kernel': randint(3,10),
            'pooling': randint(1,3),
        }

        for h in selected_hyperparam: 
            hparam[h] = randparam[h]
        self.opt['learning_rate'] = 10 ** hparam['log_lr']
        self.opt['minibatch_size'] =  int(2 ** hparam['log_batch_size'])
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


    def thresholds(self, model):
        """
        scanning thresholds for binarization of output
        """
        pass
        #WIP
        # #model_basename
        #benchmark = Accuracy(self.model, testset, tokenize=self.tokenize)
        #precision, recall, f1 = benchmark.run()