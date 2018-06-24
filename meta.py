#! .venv/bin/python
# -*- coding: utf-8 -*-
#T. Lemberger, 2018

"""smtag
Usage:
  meta.py [-f <file> -E <int> -Z <int> -R <float> -V <float> -o <str> -i <str> -a <str> -c <str> -n <str> -k <str> -p <str>]

Options:
  -f <file>, --file <file>                Namebase of dataset to import [default: test_train]
  -E <int>, --epochs <int>                Number of training epochs [default: 120]
  -Z <int>, --minibatch_size <int>        Minibatch size [default: 128]
  -R <float>, --learning_rate <float>     Learning rate [default: 0.001]
  -V <float>, --validation_fraction <float>    Fraction of the dataset that should be used as validation set during training [default: 0.2]
  -o <str>, --output_features <str>       Selected output features (use quotes if comma+space delimited) [default: geneprod]
  -i <str>, --features_as_input <str>     Features that should be added to the input (use quotes if comma+space delimited) [default: ]
  -a <str>, --overlap_features <str>      Features that should be combined by intersecting them (equivalent to AND operation) (use quotes if comma+space delimited) [default: ]
  -c <str>, --collapsed_features <str>    Features that should be collapsed into a single one (equivalent to OR operation) (use quotes if comma+space delimited) [default: ]
  -n <str>, --nf_table <str>              Number of features for each hidden layer (use quotes if comma+space delimited) [default: 8,8,8]
  -k <str>, --kernel_table <str>          Convolution kernel for each hidden layer (use quotes if comma+space delimited) [default: 6,6,6]
  -p <str>, --pool_table <str>            Pooling for each hidden layer (use quotes if comma+space delimited) [default: 2,2,2]
  
  -h --help     Show this screen.
  --version     Show version.
"""
#from docopt import docopt
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
import sys; print("sys.path=",sys.path) # for debugging
import argparse
import torch
from smtag.loader import Loader
from smtag.minibatches import Minibatches
from smtag.trainer import Trainer
from smtag.builder import SmtagModel
from smtag.importexport import export_model, load_model
from smtag.config import MODEL_DIR

if __name__ == '__main__':
    # logging.basicConfig(filename='myapp.log', level=logging.INFO)
    # logging.config.fileConfig('logging.yml')
    # logging.basicConfig(level=logging.INFO)
    #setup_logging()
    #logger = logging.getLogger(__name__)

    # READ COMMAND LINE ARGUMENTS
    #arguments = docopt(__doc__, version='0.1')
    parser = argparse.ArgumentParser(description='Top level module to manage training.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', default='test_entities_train', help='Namebase of dataset to import')
    parser.add_argument('-E' , '--epochs',  default=120, help='Number of training epochs.')
    parser.add_argument('-Z', '--minibatch_size', default=128, help='Minibatch size.')
    parser.add_argument('-R ', '--learning_rate', default=0.001, help='Learning rate.')
    parser.add_argument('-V', '--validation_fraction', default=0.2, help='Fraction of the dataset that should be used as validation set during training.')
    parser.add_argument('-o', '--output_features', default='geneprod', help='Selected output features (use quotes if comma+space delimited).')
    parser.add_argument('-i', '--features_as_input', default='', help='Features that should be added to the input (use quotes if comma+space delimited).')
    parser.add_argument('-a', '--overlap_features', default='', help='Features that should be combined by intersecting them (equivalent to AND operation) (use quotes if comma+space delimited).')
    parser.add_argument('-c', '--collapsed_features', default='', help='Features that should be collapsed into a single one (equivalent to OR operation) (use quotes if comma+space delimited).')
    parser.add_argument('-n', '--nf_table', default=[8,8,8], help='Number of features in each hidden super-layer.')
    parser.add_argument('-k', '--kernel_table', default=[6,6,6], help='Convolution kernel for each hidden layer.')
    parser.add_argument('-p', '--pool_table',  default= [2,2,2], help='Pooling for each hidden layer (use quotes if comma+space delimited).')

    arguments = vars(parser.parse_args()) # to cast as dict which is what is return by docopt in case we would use it
    print(arguments)
    # map arguments to opt to decouple command line options from internal representation
    opt = {}
    opt['namebase'] = arguments['file']
    opt['learning_rate'] = float(arguments['learning_rate'])
    opt['epochs'] = int(arguments['epochs'])
    opt['minibatch_size'] = int(arguments['minibatch_size'])
    output_features = [x.strip() for x in arguments['output_features'].split(',') if x.strip()]
    collapsed_features = [x.strip() for x in arguments['collapsed_features'].split(',') if x.strip()]
    overlap_features = [x.stip() for x in arguments['overlap_features'].split(',') if x.strip()]
    features_as_input = [x.strip() for x in arguments['features_as_input'].split(',') if x.strip()]
    nf_table = [x for x in arguments['nf_table']] # .split(',')]
    kernel_table = [x for x in arguments['kernel_table']] # .split(',')]
    pool_table = [x for x in arguments['pool_table']] # .split(',')]
    opt['selected_features'] = output_features
    opt['collapsed_features'] = collapsed_features
    opt['overlap_features'] = overlap_features
    opt['features_as_input'] = features_as_input
    opt['nf_table'] =  nf_table
    opt['pool_table'] = pool_table
    opt['kernel_table'] = kernel_table
    opt['dropout'] = 0.1
    opt['validation_fraction'] = float(arguments['validation_fraction'])
    print("\n".join(["opt[{}]={}".format(o,opt[o]) for o in opt]))

    #LOAD DATA
    ldr = Loader(opt)
    datasets = ldr.prepare_datasets(opt['namebase'])
    training_minibatches = Minibatches(datasets['train'], opt['minibatch_size'])
    validation_minibatches = Minibatches(datasets['valid'], opt['minibatch_size'])
    opt['nf_input'] = datasets['train'].nf_input
    opt['nf_output'] =  datasets['train'].nf_output
    print("input, output sizes: {}, {}".format(training_minibatches[0].output.size(), training_minibatches[0].output.size()))

    #TRAIN MODEL
    model = SmtagModel(opt)
    t = Trainer(training_minibatches, validation_minibatches, model)
    t.train()

    #SAVE MODEL
    export_model(model)
