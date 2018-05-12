"""smtag
Usage:
  meta.py [-f <file>]

Options:
  -f --file <file>     namebase of dataset to import [default: test]
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

from smtag import loader
if __name__ == '__main__':
    # logging.basicConfig(filename='myapp.log', level=logging.INFO)
    # logging.config.fileConfig('logging.yml')
    # logging.basicConfig(level=logging.INFO)
    setup_logging()
    logger = logging.getLogger(__name__)
    arguments = docopt(__doc__, version='0.1')
    ldr = loader.Loader([])
    dataset = ldr.prepare_datasets(arguments['--file'])
    minibatches = Minibatches(dataset)
