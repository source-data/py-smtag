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
from .loader import Loader
from .minibatches import Minibatches
from .trainer import Trainer
from .scanner import HyperScan
from .builder import SmtagModel
from ..common.utils import cd
from ..common.importexport import export_model, load_model
from .. import config


class Meta():

    def __init__(self, opt):
        self.opt = opt

    def _load_data(self):
        ldr = Loader(self.opt) # careful: needs to be re-initialized since batch size can vary in hyper scan
        datasets = {
            'train': ldr.prepare_datasets(os.path.join(config.data4th_dir, self.opt['namebase'], 'train')),
            'valid': ldr.prepare_datasets(os.path.join(config.data4th_dir, self.opt['namebase'], 'valid'))
        }
        return datasets

    def _prep_minibatches(self, datasets):
        training_minibatches = Minibatches(datasets['train'], self.opt['minibatch_size'])
        validation_minibatches = Minibatches(datasets['valid'], self.opt['minibatch_size'])
        self.opt['nf_input'] = datasets['train'].nf_input # dataset
        self.opt['nf_output'] =  datasets['train'].nf_output # dataset 
        print("input, output sizes: {}, {}".format(training_minibatches[0].output.size(), training_minibatches[0].output.size()))
        return training_minibatches, validation_minibatches # minibatches

    def _train(self, training_minibatches, validation_minibatches, opt):
        # check if previous model specified and load it with importmodel
        if opt['modelname']:
            model = load_model(opt['modelname'])
        else:
            model = SmtagModel(opt)
        train_loss, valid_loss, precision, recall, f1 = Trainer(training_minibatches, validation_minibatches, model).train()
        return model, {'train_loss': train_loss, 'valid_loss': valid_loss, 'precision': precision, 'recall': recall, 'f1': f1}

    def _save(self, model):
        export_model(model)

    def simple_training(self):
        datasets = self._load_data()
        training_minibatches, validation_minibatches = self._prep_minibatches(datasets)
        model, perf = self._train(training_minibatches, validation_minibatches, self.opt)
        print("final perf ({}):".format("\t".join([x for x in perf])), "\t".join(["{:.2}".format(x) for x in perf]))
        self._save(model)

    def hyper_scan(self, iterations, hyperparams, scan_name):
        datasets = self._load_data() # CAREFUL: needs to re-prepare minibatches since minibatch size is a hyperparam
        with cd(config.scans_dir):
            scan = HyperScan(self.opt, scan_name)
            for i in range(iterations):
                self.opt = scan.randopt(hyperparams) # obtain random sampling from selected hyperparam
                training_minibatches, validation_minibatches = self._prep_minibatches(datasets)
                model, perf = self._train(training_minibatches, validation_minibatches, self.opt) # perf is  dict {'train_loss': train_loss, 'valid_loss': valid_loss, 'precision': precision, 'recall': recall, 'f1': f1}
                scan.append(model, perf, self.opt, i)

def main():
    # logging.basicConfig(filename='myapp.log', level=logging.INFO)
    # logging.config.fileConfig('logging.yml')
    # logging.basicConfig(level=logging.INFO)
    #setup_logging()
    #logger = logging.getLogger(__name__)

    # READ COMMAND LINE ARGUMENTS
    #arguments = docopt(__doc__, version='0.1')
    parser = argparse.ArgumentParser(description='Top level module to manage training.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', default='demo_xml_train', help='Namebase of dataset to import')
    parser.add_argument('-E' , '--epochs',  default=200, help='Number of training epochs.')
    parser.add_argument('-Z', '--minibatch_size', default=32, help='Minibatch size.')
    parser.add_argument('-R', '--learning_rate', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('-o', '--output_features', default='geneprod', help='Selected output features (use quotes if comma+space delimited).')
    parser.add_argument('-i', '--features_as_input', default='', help='Features that should be added to the input (use quotes if comma+space delimited).')
    parser.add_argument('-a', '--overlap_features', default='', help='Features that should be combined by intersecting them (equivalent to AND operation) (use quotes if comma+space delimited).')
    parser.add_argument('-c', '--collapsed_features', default='', help='Features that should be collapsed into a single one (equivalent to OR operation) (use quotes if comma+space delimited).')
    parser.add_argument('-n', '--nf_table', default="8,8,8", help='Number of features in each hidden super-layer.')
    parser.add_argument('-k', '--kernel_table', default="6,6,6", help='Convolution kernel for each hidden layer.')
    parser.add_argument('-p', '--pool_table',  default="2,2,2", help='Pooling for each hidden layer (use quotes if comma+space delimited).')
    parser.add_argument('-w', '--working_directory', help='Specify the working directory for meta, where to read and write files to')
    parser.add_argument('-H', '--hyperparams', default='', help='Perform a scanning of the hyperparameters selected.')
    parser.add_argument('-I', '--iterations', default=25, help='Number of iterations for the hyperparameters scanning.')
    parser.add_argument('-m', '--model', default='', help='Load pre-trained model and continue training.')
    parser.add_argument('--ocr', action="store_true", help='Use as additional input features extracted by OCR from the illustration.')
    parser.add_argument('--viz', action="store_true", help='Use as additional visual features extracted from the illustration.')


    arguments = parser.parse_args()
    hyperparams = [x.strip() for x in arguments.hyperparams.split(',') if x.strip()]
    iterations = int(arguments.iterations)
    opt = {}
    opt['namebase'] = arguments.file
    opt['modelname'] = arguments.model
    opt['learning_rate'] = float(arguments.learning_rate)
    opt['epochs'] = int(arguments.epochs)
    opt['minibatch_size'] = int(arguments.minibatch_size)
    output_features = [x.strip() for x in arguments.output_features.split(',') if x.strip()]
    collapsed_features = [x.strip() for x in arguments.collapsed_features.split(',') if x.strip()]
    overlap_features = [x.strip() for x in arguments.overlap_features.split(',') if x.strip()]
    features_as_input = [x.strip() for x in arguments.features_as_input.split(',') if x.strip()]
    nf_table = [int(x.strip()) for x in arguments.nf_table.split(',')]
    kernel_table = [int(x.strip()) for x in arguments.kernel_table.split(',')]
    pool_table = [int(x.strip()) for x in arguments.pool_table.split(',')]
    opt['selected_features'] = output_features
    opt['collapsed_features'] = collapsed_features
    opt['overlap_features'] = overlap_features
    opt['features_as_input'] = features_as_input
    opt['nf_table'] =  nf_table
    opt['pool_table'] = pool_table
    opt['kernel_table'] = kernel_table
    opt['dropout'] = 0.1
    opt['use_ocr_context'] = arguments.ocr
    opt['use_viz_context'] = arguments.viz
    print("\n".join(["opt[{}]={}".format(o,opt[o]) for o in opt]))

    if arguments.working_directory:
        config.working_directory = arguments.working_directory
    with cd(config.working_directory):
        metatrainer = Meta(opt)
        if not hyperparams:
            metatrainer.simple_training()
        else:
            scan_name = 'scan_'
            scan_name += "_".join([k for k in hyperparams])
            scan_name += "_X"+str(iterations)
            metatrainer.hyper_scan(iterations, hyperparams, scan_name)

if __name__ == '__main__':
    main()
