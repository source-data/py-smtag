"""smtag
Usage:
  cli.py [-m <str> -t <str> -f <xml]

Options:
  -m <str>, --method <str>                Method to call [default: entity)
  -t <str>, --text <str>                  Text input in unicode.
  -f <str>, --format <str>                Format of the output [default: xml]
  -g <str>, --tag <str>                   XML tag to update when using the role prediction method [default: xml]
"""	

from torch import nn
import torch
from docopt import docopt
from smtag.importexport import load_model
from smtag.predictor import EntityPredictor
from smtag.config import PROD_DIR

class Concat(nn.Module):#SmtagModel?

    def __init__(self, model_list):
        super(Concat, self).__init__()
        self.model_list = []
        self.output_semantics = []
        for m in model_list:
            self.add_module('_'.join(m.output_semantics), m)
            self.output_semantics += m.output_semantics
        self.concat = lambda tensor_list: torch.cat(tensor_list, 1)

    def forward(self, x):
        y = []
        for m in self.children():
            y.append(m(x))
        return self.concat(y)

# PARSE ARGUMENTS
arguments = docopt(__doc__, version='0.1')
input_string = arguments['--text']

#LOAD MODELS
#ENTITIES
entity_models = ['geneprod.zip']
model_list = []
for filename in entity_models:
    model_list.append(load_model(filename, PROD_DIR))
#assemble into single model
entity_model = Concat(model_list)

#NON ANONYMIZED SEMANTICS
#PURE CONTEXT SEMANTICS
#BOUNDARIES


#PREDICT
p = EntityPredictor(entity_model)
ml = p.markup(input_string)
print(ml)