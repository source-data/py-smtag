#! .venv/bin/python
# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import os
#import yaml
#import logging
#import logging.config
#def setup_logging(path="logging.yaml", default_level=logging.INFO):
#    if os.path.exists(path):
#        with open(path, 'rt') as f:
#            config = yaml.safe_load(f.read())
#        logging.config.dictConfig(config)
#    else:
#        logging.basicConfig(level=default_level)
import argparse
import torch
from json import JSONEncoder
from .dataset import Data4th
from .trainer import Trainer
from .scanner import HyperScan
from .builder import SmtagModel
from ..common.utils import cd
from ..common.importexport import load_container
from ..common.options import Options
from ..common.embeddings import EMBEDDINGS
from .. import config


class Meta():

    def __init__(self, opt, production_mode=False):
        self.opt = opt
        if production_mode:
            self.trainset = Data4th(opt, ['train','valid'])
            self.validation = Data4th(opt, ['test'])
        else:
            self.trainset = Data4th(opt, ['train'])
            self.validation = Data4th(opt, ['valid'])
        self.opt.L = self.trainset.opt.L

    def _train(self, trainset, validation, opt):
        # check if previous model specified and load it with importmodel
        if opt.modelname:
            model = load_container(opt.modelname) # load pre-trained pre-existing model
        else:
            model = SmtagModel(opt)
            print(model)
        last_model, best_f1 = Trainer(trainset, validation, model).train() # best models saved to disk
        return  last_model, best_f1

    def simple_training(self):
        self._train(self.trainset, self.validation, self.opt) # models are saved to disk during training

    def hyper_scan(self, iterations, hyperparams, scan_name):
        pass
    #     with cd(config.scans_dir):
    #         scan = HyperScan(self.opt, scan_name)
    #         for i in range(iterations):
    #             self.opt = scan.randopt(hyperparams) # obtain random sampling from selected hyperparam
    #             best_model_name, best_f1 = self._train(self.trainset, self.validation, self.opt) # perf is  dict {'train_loss': train_loss, 'valid_loss': valid_loss, 'precision': precision, 'recall': recall, 'f1': f1}
    #             scan.append(best_model_name, best_f1, self.opt, i)

def main():
    parser = config.create_argument_parser_with_defaults(description='Top level module to manage training.')
    parser.add_argument('-f', '--files', default='', help='Namebase of dataset to import')
    parser.add_argument('-E' , '--epochs',  default=200, type=int, help='Number of training epochs.')
    parser.add_argument('-Z', '--minibatch_size', default=32, type=int, help='Minibatch size.')
    parser.add_argument('-R', '--learning_rate', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('-D', '--dropout_rate', default=0.1, type=float, help='Dropout rate.')
    parser.add_argument('-o', '--output_features', default='geneprod', help='Selected output features (use quotes if comma+space delimited).')
    parser.add_argument('-c', '--hidden_channels', default=32, type=int, help='Number of features in each hidden super-layer.')
    parser.add_argument('-k', '--kernel', default=7, type=int, help='Convolution kernel for each hidden layer.')
    parser.add_argument('-s', '--stride', default=1, type=int, help='Stride of the convolution.')
    parser.add_argument('-g', '--padding',  default=3, type=int, help='Padding for each hidden layer (use quotes if comma+space delimited).')
    parser.add_argument('-N', '--N_layers', default=3, type=int, help="Number of layers in the model.")
    parser.add_argument('--hyperparams', default='', help='Perform a scanning of the hyperparameters selected.')
    parser.add_argument('--iterations', default=25, type=int, help='Number of iterations for the hyperparameters scanning.')
    parser.add_argument('--production', action='store_true', help='Production mode, where train and valid are combined and test used to control for overfitting.')
    parser.add_argument('--model', default='', help='Load pre-trained model and continue training.')
    
    arguments = parser.parse_args()
    hyperparams = [x.strip() for x in arguments.hyperparams.split(',') if x.strip()]
    iterations = int(arguments.iterations)
    opt = {}
    opt['data_path_list'] = [dir.strip() for dir in arguments.files.split(',')]
    opt['namebase'] = "-".join(opt['data_path_list'])
    opt['modelname'] = arguments.model
    opt['learning_rate'] = float(arguments.learning_rate)
    opt['dropout_rate'] = float(arguments.dropout_rate)
    opt['epochs'] = arguments.epochs
    opt['minibatch_size'] = arguments.minibatch_size
    opt['selected_features'] = [x.strip() for x in arguments.output_features.split(',') if x.strip()]
    opt['N_layers'] = arguments.N_layers
    opt['hidden_channels'] = arguments.hidden_channels
    opt['kernel'] = arguments.kernel
    opt['padding'] = arguments.padding
    opt['stride'] = arguments.stride
    if config.embeddings_model:
        opt['nf_input'] = EMBEDDINGS.model.out_channels # config.nbits # WARNING: this should change when using EMBEDDINGS
    else:
        opt['nf_input'] = config.nbits
    production_mode = arguments.production
    options = Options(opt)

    metatrainer = Meta(options, production_mode)
    if not hyperparams:
        metatrainer.simple_training()
    else:
        scan_name = 'scan_'
        scan_name += "_".join([k for k in hyperparams])
        scan_name += "_X"+str(iterations)
        metatrainer.hyper_scan(iterations, hyperparams, scan_name)

if __name__ == '__main__':
    main()
