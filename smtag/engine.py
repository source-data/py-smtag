# -*- coding: utf-8 -*-
#T. Lemberger, 2018

"""smtag
Usage:
  engine.py [-m <str> -t <str> -f <str>]

Options:

  -m <str>, --method <str>                Method to call [default: smtag]
  -t <str>, --text <str>                  Text input in unicode [default: Fluorescence microcopy images of GFP-Atg5 in fibroblasts from Creb1-/- mice after bafilomycin treatment.].
  -f <str>, --format <str>                Format of the output [default: xml]
  -g <str>, --tag <str>                   XML tag to update when using the role prediction method [default: sd-tag]
"""	

from torch import nn
import torch
from docopt import docopt
from smtag.importexport import load_model
from smtag.utils import tokenize, timer
from smtag.builder import Concat
from smtag.converter import TString
from smtag.mapper import Catalogue
from smtag.binarize import Binarized
from smtag.predictor import SimplePredictor, ContextualPredictor
from smtag.serializer import Serializer
from smtag.config import PROD_DIR
from smtag.viz import Show

# maybe should be in buidler.py
class Combine(nn.Module):#SmtagModel?
    '''
    This module takes a list of SmtagModels and concatenates their output along the second dimension (feature axe).
    The output_semantics keeps track of the semantics of each module in the right order.
    '''

    def __init__(self, model_list):
        super(Combine, self).__init__()
        self.model_list = []
        self.output_semantics = []
        self.anonymize_with = []
        for model, anonymize_with in model_list:
            # need to handle empty position and return identity ?
            name = 'unet2__'+'_'.join([str(e) for e in model.output_semantics])
            self.add_module(name, model)
            #print("model.output_semantics", ", ".join([str(f) for f in model.output_semantics]))
            self.output_semantics += model.output_semantics # some of them can have > 1 output feature hence += instead of .append
            self.anonymize_with.append(anonymize_with) # what is anonymize_with is None or empty?
        self.concat = Concat(1)

    def forward(self, x):
        y_list = []
        for n, m in self.named_children():
            if n[0:7] == 'unet2__': # the child module is one of the unet models
                y_list.append(m(x))
        y = self.concat(y_list)
        return y

class Connector(nn.Module):
    '''
    A module to connect A to B such that the output features of A (output_semantics) match the input features of B (input_semantics).
    Usage example:
        rewire = Connector(self.models['entity'].output_semantics, self.models['context'].anonymize_with)
    '''
    def __init__(self, output_semantics, input_semantics): 
        super(Connector, self).__init__()
        #get indices of output channels (of the source module) in the order require by input semantics (of the receiving module).
        print("input_semantics", " ".join([str(c) for c in input_semantics]))
        print("output_semantics", " ".join([str(c) for c in output_semantics]))
        print(Catalogue.GENEPROD.my_index(input_semantics))
        # features should match by type or by role or by both?
        # implement eq, eqByRole, eqByType?
        matches = list(filter(None, [concept.my_index(output_semantics) for concept in input_semantics])) # finds the position of the required input concept in the list of output concepts
        print("matches ", matches)
        self.indices = matches

    def forward(self, x):
        '''
        returns the tensor where second dimension (channels) of x are reordered to match the needs of the downstream B module.
        '''
        return x[ : , self.indices, : ]


class SmtagEngine:

    # @timer
    def __init__(self, cartridge={}):
        #change this to accept a 'cartridge' that descibes which models to load
        if cartridge:
            self.cartridge = cartridge
        else:
            self.cartridge = {
                # '<model-family>' : [(<model>, <features that needs to be anonimized>), ...]
                'entity': [
                    (load_model('small_molecule.zip', PROD_DIR), ''),
                    (load_model('geneprod.zip', PROD_DIR), ''),
                    (load_model('subcellular.zip', PROD_DIR), ''),
                    (load_model('cell.zip', PROD_DIR), ''),
                    (load_model('tissue.zip', PROD_DIR), ''),
                    (load_model('organism.zip', PROD_DIR), ''),
                    (load_model('exp_assay.zip', PROD_DIR), '')
                ],
                'only_once': [
                    (load_model('reporter_geneprod.zip', PROD_DIR), '')
                ],
                'context': [
                    (load_model('causality_geneprod.zip', PROD_DIR), 'geneprod')
                ], 
                'panelizer': [
                    (load_model('panel_start.zip', PROD_DIR), '')
                ]
            }
        self.models = {}
        for model_family in self.cartridge:
            self.models[model_family] = Combine([(model, Catalogue.from_label(anonymize_with)) for model, anonymize_with in self.cartridge[model_family]])

    # @timer
    def __entity(self, input_string):
        input_t_string = TString(input_string)
        p = SimplePredictor(self.models['entity'])
        binarized = p.pred_binarized(input_t_string, self.models['entity'].output_semantics)
        return binarized

    def entity(self, input_string):
        return self.serialize(self.__entity(input_string))

    # @timer
    def __entity_and_context(self, input_string):

        input_t_string = TString(input_string)
        
        entity_p = SimplePredictor(self.models['entity'])
        binarized = entity_p.pred_binarized(input_t_string, self.models['entity'].output_semantics)

        # select and order the predicted marks to be fed to the second context_p semantics from context model.
        rewire = Connector(self.models['entity'].output_semantics, self.models['context'].anonymize_with)
        marks = rewire.forward(binarized.marks)

        context_p = ContextualPredictor(self.models['context'])
        context_binarized = context_p.pred_binarized(input_string, marks, self.models['context'].output_semantics)
        
        #concatenate entity and output_semantics before calling Serializer()
        binarized.cat_(context_binarized)
        return binarized

    def tag(self, input_string):
        return self.serialize(self.__entity_and_context(input_string))

    # @timer
    def __all(self, input_string):
        
        input_t_string = TString(input_string)
        print(input_t_string)
        #PREDICT PANELS
        panel_p = SimplePredictor(self.models['panelizer'])
        binarized_panels = panel_p.pred_binarized(input_t_string, self.models['panelizer'].output_semantics)

        # PREDICT ENTITIES
        entity_p = SimplePredictor(self.models['entity'])
        binarized_entities = entity_p.pred_binarized(input_t_string, self.models['entity'].output_semantics)
        print("\n0: binarized.marks after entity"); Show.print_pretty(binarized_entities.marks)
        print("output semantics: ", "; ".join([str(e) for e in binarized_entities.output_semantics]))

        cumulated_output = binarized_entities.clone()
        cumulated_output.cat_(binarized_panels)
        print("\n1: cumulated_output.marks after panel and entity"); Show.print_pretty(cumulated_output.marks)
        print("output semantics: ", "; ".join([str(e) for e in cumulated_output.output_semantics]))

        # PREDICT REPORTERS
        reporter_p = SimplePredictor(self.models['only_once'])
        binarized_reporter = reporter_p.pred_binarized(input_t_string, self.models['only_once'].output_semantics)
        print("\n2: binarized_reporter.marks");Show.print_pretty(binarized_reporter.marks)
        print("output semantics: ", "; ".join([str(e) for e in binarized_reporter.output_semantics]))

        # there should be as many reporter model slots, even if empty, as entities are predicted.
        binarized_entities.erase_(binarized_reporter) # will it erase itself? need to assume output is 1 channel only
        print("\n3: binarized_entities.marks after erase_(reporter)"); Show.print_pretty(binarized_entities.marks)
        print("output semantics: ", "; ".join([str(e) for e in binarized_entities.output_semantics]))

        cumulated_output.cat_(binarized_reporter) # add reporter prediction to output features
        print("\n4: cumulated_output.marks after cat_(reporter)"); Show.print_pretty(cumulated_output.marks)
        print("output semantics: ", "; ".join([str(e) for e in cumulated_output.output_semantics]))

        # select and match by type the predicted entity marks to be fed to the second context_p semantics from context model.
        rewire = Connector(self.models['entity'].output_semantics, self.models['context'].anonymize_with)
        anonymization_marks = rewire.forward(binarized_entities.marks) # should not include reporter marks; 
        print("\n5: anonymization_marks"); Show.print_pretty(anonymization_marks)

        # PREDICT ROLES ON NON REPORTER ENTITIES
        context_p = ContextualPredictor(self.models['context'])
        context_binarized = context_p.pred_binarized(input_t_string, anonymization_marks, self.models['context'].output_semantics)
        print("\n6: context_binarized"); Show.print_pretty(context_binarized.marks)

        #concatenate entity and output_semantics before calling Serializer()
        cumulated_output.cat_(context_binarized)
        print("\n7: final cumulated_output.marks");Show.print_pretty(cumulated_output.marks)
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
    print(arguments)
    input_string = arguments['--text']
    method = arguments['--method']
    engine = SmtagEngine()

    if method == 'smtag':
        print(engine.smtag(input_string))
    elif method == 'panelize':
        print(engine.panelizer(input_string))
    elif method == 'tag':
        print(engine.tag(input_string))
    elif method == 'entity':
        print(engine.entity(input_string))
    else:
        print("unknown method {}".format(method))
