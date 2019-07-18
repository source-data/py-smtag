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
from ..common.importexport import export_model, load_model
from ..common.options import Options
from ..common.embeddings import EMBEDDINGS
from .. import config


class Meta():

    def __init__(self, opt):
        self.opt = opt
        self.trainset = Data4th(opt, 'train')
        self.validation = Data4th(opt, 'valid')
        self.opt.L = self.trainset.opt.L
        
        
    def _train(self, trainset, validation, opt):
        # check if previous model specified and load it with importmodel
        if opt.modelname:
            model = load_model(opt.modelname)
        else:
            model = SmtagModel(opt)
            print(model)
        train_loss, valid_loss, precision, recall, f1 = Trainer(trainset, validation, model).train()
        return model, {'train_loss': train_loss, 'valid_loss': valid_loss, 'precision': precision, 'recall': recall, 'f1': f1}

    def _save(self, model):
        export_model(model)

    def simple_training(self):
        model, perf = self._train(self.trainset, self.validation, self.opt)
        print("final perf ({}):".format("\t".join([x for x in perf])), "\t".join(["{:.2}".format(x) for x in perf]))
        self._save(model)

    def hyper_scan(self, iterations, hyperparams, scan_name):
        with cd(config.scans_dir):
            scan = HyperScan(self.opt, scan_name)
            for i in range(iterations):
                self.opt = scan.randopt(hyperparams) # obtain random sampling from selected hyperparam
                model, perf = self._train(self.trainset, self.validation, self.opt) # perf is  dict {'train_loss': train_loss, 'valid_loss': valid_loss, 'precision': precision, 'recall': recall, 'f1': f1}
                scan.append(model, perf, self.opt, i)

def main():
    # logging.basicConfig(filename='myapp.log', level=logging.INFO)
    # logging.config.fileConfig('logging.yml')
    # logging.basicConfig(level=logging.INFO)
    #setup_logging()
    #logger = logging.getLogger(__name__)

    # READ COMMAND LINE ARGUMENTS
    #arguments = docopt(__doc__, version='0.1')
    parser = config.create_argument_parser_with_defaults(description='Top level module to manage training.')
    parser.add_argument('-f', '--files', default='demo_xml_train', help='Namebase of dataset to import')
    parser.add_argument('-E' , '--epochs',  default=200, help='Number of training epochs.')
    parser.add_argument('-Z', '--minibatch_size', default=32, help='Minibatch size.')
    parser.add_argument('-R', '--learning_rate', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('-D', '--dropout_rate', default=0.1, type=float, help='Dropout rate.')
    parser.add_argument('-S', '--no_skip', action='store_true', help="Use this option to __deactivate__ skip links in unet2 model.")
    parser.add_argument('-o', '--output_features', default='geneprod', help='Selected output features (use quotes if comma+space delimited).')
    parser.add_argument('-n', '--nf_table', default="8,8,8", help='Number of features in each hidden super-layer.')
    parser.add_argument('-k', '--kernel_table', default="7,7,7", help='Convolution kernel for each hidden layer.')
    # parser.add_argument('-p', '--pool_table',  default="2,2,2", help='Pooling for each hidden layer (use quotes if comma+space delimited).')
    parser.add_argument('-g', '--padding_table',  default="3,3,3", help='Padding for each hidden layer (use quotes if comma+space delimited).')
    parser.add_argument('-H', '--hyperparams', default='', help='Perform a scanning of the hyperparameters selected.')
    parser.add_argument('-I', '--iterations', default=25, help='Number of iterations for the hyperparameters scanning.')
    parser.add_argument('-m', '--model', default='', help='Load pre-trained model and continue training.')
    parser.add_argument('--ocrxy', action="store_true", help='Use as additional input position and orientation of words extracted by OCR from the illustration.')
    parser.add_argument('--ocr1', action="store_true", help='Use as additional presence of words extracted by OCR from the illustration.')
    parser.add_argument('--ocr2', action="store_true", help='Use as additional input orientation of words extracted by OCR from the illustration.')
    parser.add_argument('-V', '--viz_table', default="", help='Use as additional visual features extracted from the illustration.')

    arguments = parser.parse_args()
    hyperparams = [x.strip() for x in arguments.hyperparams.split(',') if x.strip()]
    iterations = int(arguments.iterations)
    opt = {}
    opt['data_path_list'] = [dir.strip() for dir in arguments.files.split(',')]
    opt['namebase'] = "-".join(opt['data_path_list'])
    opt['modelname'] = arguments.model
    opt['learning_rate'] = float(arguments.learning_rate)
    opt['dropout'] = float(arguments.dropout_rate)
    opt['skip'] = not arguments.no_skip
    opt['epochs'] = int(arguments.epochs)
    opt['minibatch_size'] = int(arguments.minibatch_size)
    opt['selected_features'] = [x.strip() for x in arguments.output_features.split(',') if x.strip()]
    opt['nf_table'] = [int(x.strip()) for x in arguments.nf_table.split(',')]
    opt['kernel_table'] = [int(x.strip()) for x in arguments.kernel_table.split(',')]
    # opt['pool_table'] = [int(x.strip()) for x in arguments.pool_table.split(',')]
    opt['padding_table'] = [int(x.strip()) for x in arguments.padding_table.split(',')]
    if arguments.viz_table:
        opt['viz_context_table'] = [int(x.strip()) for x in arguments.viz_table.split(',')]
    else:
        opt['viz_context_table'] = ''
    if arguments.ocrxy:
        opt['use_ocr_context'] = 'ocrxy'
    elif arguments.ocr1:
        opt['use_ocr_context'] = 'ocr1'
    elif arguments.ocr2:
        opt['use_ocr_context'] = 'ocr2'
    else:
        opt['use_ocr_context'] = ''
    opt['nf_input'] = EMBEDDINGS.out_channels # config.nbits # WARNING: this should change when using EMBEDDINGS
    options = Options(opt)
    # print(options)c

    metatrainer = Meta(options)
    if not hyperparams:
        metatrainer.simple_training()
    else:
        scan_name = 'scan_'
        scan_name += "_".join([k for k in hyperparams])
        scan_name += "_X"+str(iterations)
        metatrainer.hyper_scan(iterations, hyperparams, scan_name)

if __name__ == '__main__':
    main()
