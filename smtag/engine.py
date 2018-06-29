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

from torch import nn
import torch
from docopt import docopt
from smtag.importexport import load_model
from smtag.utils import tokenize, timer
from smtag.builder import Concat
from smtag.converter import TString
from smtag.binarize import Binarized
from smtag.predictor import SimplePredictor, ContextualPredictor
from smtag.serializer import Serializer
from smtag.config import PROD_DIR
from smtag.viz import Show

# maybe should be in buidler.py
class Combine(nn.Module):#SmtagModel?

    def __init__(self, model_list):
        super(Combine, self).__init__()
        self.model_list = []
        self.output_semantics = []
        self.anonymize_with = []
        for model, anonymize_with in model_list:
            # need to handle empty position and return identity
            name = 'unet2__'+'_'.join([str(e) for e in model.output_semantics])
            self.add_module(name, model)
            print("model.output_semantics", model.output_semantics)
            self.output_semantics += model.output_semantics # some of them can have > 1 output feature hence += instead of .append
            self.anonymize_with.append(anonymize_with)
        self.concat = Concat(1)

    def forward(self, x):
        y_list = []
        for n, m in self.named_children():
            if n[0:7] == 'unet2__': # the child module is one of the unet models
                y_list.append(m(x))
        y = self.concat(y_list)
        return y

class Connector(nn.Module):
    def __init__(self, output_semantics, input_semantics):
        super(Connector, self).__init__()
        self.output_semantics = output_semantics
        self.input_semantics = input_semantics
        #get indices of output channels (of the source module) in the order require by input semantics (of the receiving module).
        self.indices = [input_semantics.index(input) for input in input_semantics] # finds the position of the required input concept in the list of output concepts

    def forward(self, x):
        #returns the tensor where second dimension (channels) are reordered appropriately
        return x[ : , self.indices, : ]


class SmtagEngine:

    def __init__(self, cartridge={}):
        #change this to accept a 'cartridge' that descibes which models to load
        if cartridge:
            self.cartridge = cartridge
        else:
            self.cartridge = {
                'entities': [
                    (load_model('geneprod.zip'), ''),
                    (load_model('small_molecule.zip'), '')
                ],
                'only_once': [
                    (load_model('reporter_geneprod.zip'), '')
                ],
                'context': [
                    (load_model('causality_entities.zip'), 'geneprod'),
                    (load_model('causality_small_mol.zip'), 'small_mol')
                ], 
                'panelizer': [
                    (load_model('panel_start.zip'), '')
                ]
            }
        self.models = {}
        for model_family in self.cartridge:
            self.models[model_family] = Combine([(model, anonymize_with) for model, anonymize_with in cartridge[model_family]])

    @timer
    def __entities(self, input_string):
        input_t_string = TString(input_string)
        p = SimplePredictor(self.models['entity'])
        binarized = p.pred_binarized(input_t_string, self.models['entity'].output_semantics)
        return binarized

    def entity(self, input_string):
        return self.serialize(self.__entities(input_string))

    @timer
    def __entity_and_context(self, input_string):

        input_t_string = TString(input_string)
        
        entity_p = SimplePredictor(self.model['entity'])
        binarized = entity_p.pred_binarized(input_t_string, self.models['entity'].output_semantics)

        # select and order the predicted marks to be fed to the second context_p semantics from context model.
        rewire = Connector(self.models['entity'].output_semantics, self.model['context'].anonymize_with)
        marks = rewire.forward(binarized.marks)

        context_p = ContextualPredictor(self.models['context'])
        context_binarized = context_p.pred_binarized(input_string, marks, self.model['context'].output_semantics)
        
        #concatenate entity and output_semantics before calling Serializer()
        binarized.cat_(context_binarized)
        return binarized

    def tag(self, input_string):
        return self.serialize(self.__entity_and_context(input_string))

    def __all(self, input_string):
        
        input_t_string = TString(input_string)

        #PREDICT PANELS
        panel_p = SimplePredictor(self.models['panelizer'])
        binarized_panels = panel_p.pred_binarized(input_t_string, self.models['panelizer'].output_semantics)

        # PREDICT ENTITIES
        entity_p = SimplePredictor(self.models['entity'])
        binarized_entities = entity_p.pred_binarized(input_t_string, self.models['entity'].output_semantics)
        print("binarized.marks after entity"); Show.print_pretty(binarized_entities.marks)
        print("output semantics: ", "; ".join([str(e) for e in binarized_entities.output_semantics]))

        cumulated_output = binarized_entities.clone()
        cumulated_output.cat_(binarized_panels)
        print("1: cumulated_output.marks after panel and entity"); Show.print_pretty(cumulated_output.marks)
        print("output semantics: ", "; ".join([str(e) for e in cumulated_output.output_semantics]))

        # PREDICT REPORTERS
        reporter_p = SimplePredictor(self.models['only_once'])
        binarized_reporter = reporter_p.pred_binarized(input_t_string, self.models['only_once'].output_semantics)
        print("2: binarized_reporter.marks");Show.print_pretty(binarized_reporter.marks)
        print("output semantics: ", "; ".join([str(e) for e in binarized_reporter.output_semantics]))

        # there should be as many reporter model slots, even if empty, as entities are predicted.
        binarized_entities.erase_(binarized_reporter) # will it erase itself? need to assume output is 1 channel only
        print("3: binarized_entities.marks after erase_(reporter)"); Show.print_pretty(binarized_entities.marks)
        print("output semantics: ", "; ".join([str(e) for e in binarized_entities.output_semantics]))

        cumulated_output.cat_(binarized_reporter) # add reporter prediction to output features
        print("4: cumulated_output.marks after cat_(reporter)"); Show.print_pretty(cumulated_output.marks)
        print("output semantics: ", "; ".join([str(e) for e in cumulated_output.output_semantics]))

        # select and order the predicted entity marks to be fed to the second context_p semantics from context model.
        rewire = Connector(self.models['entity'].output_semantics, self.models['context'].anonymize_with)
        marks = rewire.forward(binarized_entities.marks) # should not include reporter marks; need to test
        print("5: marks"); Show.print_pretty(marks)

        # PREDICT ROLES ON NON REPORTER ENTITIES
        context_p = ContextualPredictor(self.models['context'])
        context_binarized = context_p.pred_binarized(input_t_string, marks, self.models['context'].output_semantics)
        print("6: context_binarized"); Show.print_pretty(context_binarized.marks)

        #concatenate entity and output_semantics before calling Serializer()
        cumulated_output.cat_(context_binarized)
        print("7: final cumulated_output.marks");Show.print_pretty(cumulated_output.marks)
        print("output semantics: ", "; ".join([str(e) for e in cumulated_output.output_semantics]))
        
        return cumulated_output

    def smtag(self, input_string):
        return self.serialize(self.__all(input_string))
    
    def serialize(self, output):
        ml = Serializer().serialize(output)
        return ml[0]
    
    def add_roles(self, input_xml):
        pass

    def __panels(self, input_string):
        input_t_string = TString(input_string)
        p = SimplePredictor(self.models['panelizer'])
        binarized = p.pred_binarized(input_t_string, self.models['panelizer'].output_semantics)
        return binarized

    def panelizer(self, input_string):
        return self.serialize(self.__panels(input_string))

if __name__ == "__main__":
    # PARSE ARGUMENTS
    arguments = docopt(__doc__, version='0.1')
    input_string = arguments['--text']

    print(SmtagEngine().entities(input_string))
