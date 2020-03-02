# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import os
import math
from random import randint, uniform
from copy import deepcopy
from datetime import datetime
from typing import List
from .builder import HyperparametersSmtagModel
from ..common.utils import cd

NL = "\n"
SEP= "\t"

class HyperScan():

    def __init__(self, hp: HyperparametersSmtagModel, dir_name: str):
        """
        Hyperscan will create the following dir hierarchy:
        scans/
            scan_<scan_name_1>/
                models/
                    scanned_model_1.zip
                    scanned_model_2.zip
                    ...
                scanned_perf.csv
            scan_<scan_name_1>/
                models/
                    ...
                scanned_perf.scv
        """

        self.hp = hp

        timestamp = datetime.now().isoformat("-",timespec='minutes').replace(":", "-") # dir to save scan results
        self.dir_name = dir_name + "_" + timestamp 
        if not os.path.isdir(self.dir_name):
            os.mkdir(self.dir_name)
            os.chmod(self.dir_name, mode=0o777)
        self.perf_path = os.path.join(self.dir_name, 'scanned_perf.csv')
        with cd(self.dir_name):
            if not os.path.isdir('models'):
                os.mkdir('models')
                os.chmod('models', mode=0o777)
        self.scanned_models_path = os.path.join(self.dir_name, 'models')


    def append(self, model_name, perf, opt, id):
        perf['best model'] = perf['best_model_name']
        perf['f1'] = perf['f1']
        self.append_to_csv(perf, opt, self.perf_path)

    @staticmethod
    def append_to_csv(row, opt, mypath):
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
        os.chmod(mypath, mode=0o777)


    def randopt(self, selected_hyperparam : List = ['log_lr']) -> HyperparametersSmtagModel:
        """
        Model and training hyperparameter are sampled randomly given the specified ranges.

        Args: 
            selected_hyperparamters (str) from 'log_lr' (-3) | 'log_batch_size' (5) | 'depth' (3) | 'nf' (8) | 'kernel' (6) | 'pooling' (2)

        Returns:
            dict where selected hyperparameters where randomly sampled
        """

        hparam = deepcopy(self.hp)
        #randomly sampling hyperparameters
        randparam = {
            'log_lr': uniform(-4, -1),
            'log_batch_size': uniform(4, 8),
            'N_layers': randint(1,4),
            'log_hidden_channels': randint(3, 6),
        }

        for h in selected_hyperparam: 
            hparam[h] = randparam[h]
        hp = deepcopy(self.hp)
        hp.learning_rate = 10 ** hparam['log_lr']
        self.opt['minibatch_size'] =  int(2 ** hparam['log_batch_size'])
        nf_table = []
        kernel_table = []
        pool_table =[]
        for _ in range(hparam['depth']):
            nf_table.append(2 ** hparam['log_nf'])
            kernel_table.append(hparam['kernel'])
            pool_table.append(hparam['pooling'])
        self.opt['nf_table'] = nf_table
        self.opt['kernel_table'] = kernel_table
        self.opt['pool_table'] = pool_table

        return self.opt