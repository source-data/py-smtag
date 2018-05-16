"""smtag
Usage:
  meta.py [-f <file> -Z <int> -E <int> -R <float>]

Options:
  -f <file>, --file <file>                Namebase of dataset to import [default: test_train]
  -E <int>, --epochs <int>                Number of training epochs [default: 120]
  -Z <int>, --minibatch_size <int>        Minibatch size [default: 128]
  -R <float> , --learning_rate <float>    Learning rate [default: 0.001]
  -h --help     Show this screen.
  --version     Show version.
"""
from docopt import docopt


import os
import yaml
import logging
import logging.config
def setup_logging(path="logging.yaml", default_level=logging.INFO):
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

from smtag.loader import Loader
from smtag.minibatches import Minibatches
from smtag.trainer import Trainer
from smtag.builder import Builder


if __name__ == '__main__':
    # logging.basicConfig(filename='myapp.log', level=logging.INFO)
    # logging.config.fileConfig('logging.yml')
    # logging.basicConfig(level=logging.INFO)
    setup_logging()
    logger = logging.getLogger(__name__)
    arguments = docopt(__doc__, version='0.1')
    #map arguments to opt to decouple command line options from internal representation
    opt = {}
    opt['namebase'] = arguments['--file']
    opt['learning_rate'] = float(arguments['--learning_rate'])
    opt['epochs'] = int(arguments['--epochs'])
    opt['minibatch_size'] = int(arguments['--minibatch_size'])
    opt['selected_features'] = ['gene','protein']
    opt['nf_table'] =  [8, 8, 8]
    opt['pool_table'] = [2, 2, 2]
    opt['kernel_table'] = [6, 6, 6]
    opt['dropout'] = 0.1
    opt['autoencode'] = False
    print("; ".join(opt))
    ldr = Loader(opt['selected_features'])
    datasets = ldr.prepare_datasets(opt['namebase'])
    training_minibatches = Minibatches(datasets['train'], opt['minibatch_size'])
    validation_minibatches = Minibatches(datasets['valid'], opt['minibatch_size'])
    opt['nf_input'] = datasets['train'].nf_input
    opt['nf_output'] =  datasets['train'].nf_output
    logger.info(f"input, output sizes: {training_minibatches[0].output.size()}, {training_minibatches[0].output.size()}")
    model = Builder(opt).model
    t = Trainer(training_minibatches, validation_minibatches, model)
    t.train(opt)
