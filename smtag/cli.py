# -*- coding: utf-8 -*-
#T. Lemberger, 2018

"""smtag
Usage:
  cli.py [-m <str> -t <str> -f <xml]

Options:
  -m <str>, --method <str>                Method to call [default: entity)
  -t <str>, --text <str>                  Text input in unicode [default: "fluorescent images of 200‐cell‐stage embryos from the indicated strains stained by both anti‐SEPA‐1 and anti‐LGG‐1 antibody"].
  -f <str>, --format <str>                Format of the output [default: xml]
  -g <str>, --tag <str>                   XML tag to update when using the role prediction method [default: xml]
"""	

import time
from torch import nn
import torch
from docopt import docopt
from smtag.importexport import load_model
from smtag.builder import Concat
from smtag.predictor import EntityPredictor
from smtag.config import PROD_DIR

#did not work 
#class Combine(nn.Module):#SmtagModel?
#
#    def __init__(self, model_list):
#        super(Combine, self).__init__()
#        self.model_list = []
#        self.output_semantics = []
#        for m in model_list:
#            self.add_module('_'.join(m.output_semantics), m)
#            self.output_semantics += m.output_semantics
#        self.concat = Concat(1)
#
#    def forward(self, x):
#        y_list = [m(x) for _, m in self.named_modules()]
#        return self.concat(y_list)

class Combine(nn.Module):

    def __init__(self, module_dict):
        super(Combine, self).__init__()
        #self.small_molecule = module_dict['small_molecule']
        self.output_semantics = []
        self.geneprod = module_dict['geneprod']
        self.output_semantics += self.geneprod.output_semantics
        self.concat = Concat(1)

    def forward(self, x):
        y_list = []
        #y_list.append(self.small_molecular(x))
        y_list.append(self.geneprod(x))
        return (self.concat(y_list))

# PARSE ARGUMENTS
arguments = docopt(__doc__, version='0.1')
input_string = arguments['--text']

#LOAD MODELS
#ENTITIES
model_list = {
    'geneprod': load_model('geneprod.zip', PROD_DIR)
}
entity_model = Combine(model_list)
#entity_model = load_model('geneprod.zip', PROD_DIR) # for debugging

#NON ANONYMIZED SEMANTICS

#PURE CONTEXT SEMANTICS

#BOUNDARIES

#PREDICT
p = EntityPredictor(entity_model)
start_time = time.time()
ml = p.markup(input_string)
end_time = time.time()
print(ml[0])
prediction_time = end_time - start_time
print("Prediction time: {0:.3f}s".format(prediction_time))