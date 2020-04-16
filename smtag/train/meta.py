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
from .builder import SmtagModel, HyperparemetersSmtagModel
from ..common.utils import cd
from ..common.importexport import load_smtag_model
from ..common.embeddings import EMBEDDINGS
from .. import config


class Meta():

    def __init__(self, hp: HyperparemetersSmtagModel, production_mode=False):
        self.hp = hp
        if production_mode:
            self.trainset = Data4th(hp, ['train','valid'])
            self.validation = Data4th(hp, ['test'])
        else:
            self.trainset = Data4th(hp, ['train'])
            self.validation = Data4th(hp, ['valid'])

    def _train(self, trainset: Data4th, validation: Data4th, hp: HyperparemetersSmtagModel):
        # check if previous model specified and load it with importmodel
        if hp.modelname:
            model = load_smtag_model(hp.modelname, config.model_dir) # load pre-trained pre-existing model
        else:
            model = SmtagModel(hp)
            print(model)
        model, model_name, precision, recall, f1, avg_validation_loss = Trainer(trainset, validation, model).train() # best models saved to disk
        return  model, model_name, precision, recall, f1, avg_validation_loss

    def simple_training(self):
        self._train(self.trainset, self.validation, self.hp) # models are saved to disk during training

    def hyper_scan(self, iterations, scan_params, scan_name):
        scan = HyperScan(self.hp, config.scans_dir, scan_name, scan_params)
        for i in range(iterations):
            hp = scan.randhp() # obtain random sampling from selected hyperparam
            model, model_path, precision, recall, f1, avg_validation_loss = self._train(self.trainset, self.validation, hp) # perf is  dict {'train_loss': train_loss, 'valid_loss': valid_loss, 'precision': precision, 'recall': recall, 'f1': f1}
            scan.append_to_csv({'f1': f1, 'avg_valid_loss': avg_validation_loss, 'model_path': model_path}, hp, i)

def main():
    parser = config.create_argument_parser_with_defaults(description='Top level module to manage training.')
    parser.add_argument('-f', '--files', default='', help='Namebase of dataset to import')
    parser.add_argument('-E' , '--epochs',  default=200, type=int, help='Number of training epochs.')
    parser.add_argument('-Z', '--minibatch_size', default=32, type=int, help='Minibatch size.')
    parser.add_argument('-R', '--learning_rate', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('-D', '--dropout_rate', default=0.1, type=float, help='Dropout rate.')
    parser.add_argument('-o', '--output_features', default='geneprod', help='Selected output features (use quotes if comma+space delimited).')
    parser.add_argument('-c', '--hidden_channels', default=32, type=int, help='Number of features in each hidden super-layer.')
    parser.add_argument('-n', '--nf_table', default="32,32,32", help='Number of features in each hidden super-layer.')
    parser.add_argument('-s', '--stride_table', default="1,1,1", help='Strides in hidden super-layer.')
    parser.add_argument('-k', '--kernel_table', default="7,7,7", help='Convolution kernel for each hidden layer.')
    parser.add_argument('--padding', default="0", help='Padding at each hidden layer.')
    parser.add_argument('--no_pool', action='store_true', help='Pooling for each hidden layer (use quotes if comma+space delimited).')
    parser.add_argument('-g', '--padding_table',  default="3,3,3", help='Padding for each hidden layer (use quotes if comma+space delimited).')
    parser.add_argument('--hyperscan', default='', nargs='+', choices=['learning_rate', 'minibatch_size','N_layers', 'hidden_channels'], help="Perform a scanning of the selected hyperparameters (learning_rate' | 'minibatch_size' | 'N_layers'| 'hidden_channels').")
    parser.add_argument('--iterations', default=25, type=int, help='Number of iterations for the hyperparameters scanning.')
    parser.add_argument('--production', action='store_true', help='Production mode, where train and valid are combined and test used to control for overfitting.')
    parser.add_argument('--model', default='', help='Load pre-trained model and continue training.')
 
    arguments = parser.parse_args()
    hyperscan = [x.strip() for x in arguments.hyperscan]
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
    opt['hidden_channels'] = arguments.hidden_channels
    opt['nf_table'] = [int(x.strip()) for x in arguments.nf_table.split(',')]
    opt['kernel_table'] = [int(x.strip()) for x in arguments.kernel_table.split(',')]
    opt['stride_table'] = [int(x.strip()) for x in arguments.stride_table.split(',')]
    opt['pool'] = not arguments.no_pool
    opt['padding'] = arguments.padding
 
    if config.embeddings_model:
        opt['nf_input'] = EMBEDDINGS.model.hp.out_channels
    else:
        opt['nf_input'] = config.nbits
    production_mode = arguments.production

    # create the Hyperparameter object from the command line options
    hp = HyperparemetersSmtagModel(opt)

    metatrainer = Meta(hp, production_mode)
    if not hyperscan:
        metatrainer.simple_training()
    else:
        scan_name = 'scan_'
        scan_name += "_".join([k for k in hyperscan])
        scan_name += "_X"+str(iterations)
        metatrainer.hyper_scan(iterations, hyperscan, scan_name)

if __name__ == '__main__':
    main()
