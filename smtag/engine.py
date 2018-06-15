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
from smtag.utils import tokenize
from smtag.builder import Concat
from smtag.converter import TString
from smtag.binarize import Binarized
from smtag.predictor import EntityPredictor, SemanticsFromContextPredictor
from smtag.serializer import Serializer
from smtag.config import PROD_DIR

#did not work 
class Combine(nn.Module):#SmtagModel?

    def __init__(self, model_list):
        super(Combine, self).__init__()
        self.model_list = []
        self.output_semantics = []
        self.anonymize_with = []
        for model, anonymize_with in model_list:
            name = 'U__'+'_'.join(model.output_semantics)
            self.add_module(name, model)
            self.output_semantics += model.output_semantics # some of them can have > 1 output feature hence += instead of .append
            self.anonymize_with.append(anonymize_with)
        self.concat = Concat(1)

    def forward(self, x):
        y_list = []
        for n, m in self.named_children():
            if n[0:3]=='U__':
                y_list.append(m(x))
        y = self.concat(y_list)
        return y

class Combine2(nn.Module):

    def __init__(self, module_dict):
        super(Combine2, self).__init__()
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

class Connector(nn.Module):
    def __init__(self, output_semantics, input_semantics):
        super(Connector, self).__init__()
        self.output_semantics = output_semantics
        self.input_semantics = input_semantics
        #get indices of output channels (of the source module) in the order require by input semantics (of the receiving module).
        self.indices = []
        for input in input_semantics:
            self.indices.append(input_semantics.index(input)) # finds the position of the required input concept in the list of output concepts

    def forward (self, x):
        #returns the tensor where second dimension (channels) are reordered appropriately
        return x[ : , self.indices, : ]


def timer(f):
    
    def t(*args):
        start_time = time.time()
        output = f(*args)
        end_time = time.time()
        delta_t = end_time - start_time
        print("Exec time for method '{}': {:.3f}s".format(f.__name__, delta_t))
        return output
    return t

class SmtagEngine:

    def __init__(self, dir=PROD_DIR):
        
        entity_model_list = [
            (load_model('geneprod.zip', dir), [])
        ]
        self.entity_model = Combine(entity_model_list)
        context_model_list = [
            (load_model('causality_geneprod.zip', PROD_DIR), 'geneprod') # the model_list contains the model and the expected concept that will be tagged and need anonymization
        ]
        self.context_model = Combine(context_model_list)

    @timer
    def entities(self, input_string):
        p = EntityPredictor(self.entity_model)
        ml = p.markup(input_string)
        return ml[0]
    
    @timer
    def entity_and_context(self, input_string):
        input = TString(input_string)
        entity_p = EntityPredictor(self.entity_model)
        prediction_entity = entity_p.forward(input)
        entity_binarized = Binarized([input_string], prediction_entity, self.entity_model.output_semantics)
        tokenized = tokenize(input_string)
        entity_binarized.binarize_with_token([tokenized])
        # select and order the predicted marks to be fed to the second context_p semantics from context model.
        rewire = Connector(self.entity_model.output_semantics, self.context_model.anonymize_with)
        marks = rewire.forward(entity_binarized.marks)
        context_p = SemanticsFromContextPredictor(self.context_model)
        prediction_roles = context_p.forward(input, marks) #context_p takes care of anonymization per feature
        #concatenate entity and output_semantics before calling Binarized()
        combined_pred = torch.cat([prediction_entity, prediction_roles], 1)
        combined_output_semantics = self.entity_model.output_semantics + self.context_model.output_semantics
        context_binarized = Binarized([input_string], combined_pred, combined_output_semantics)
        context_binarized.binarize_with_token([tokenized])
        ml = Serializer().serialize(context_binarized)
        return ml
    
    def add_roles(self, input_xml):
        pass

    def panels(self, input_string):
        pass
    
    def smtag(self, input_string):
        pass


# PARSE ARGUMENTS
arguments = docopt(__doc__, version='0.1')
input_string = arguments['--text']

print(SmtagEngine().entities(input_string))
