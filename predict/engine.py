# -*- coding: utf-8 -*-
#T. Lemberger, 2018

"""
SmartTag semantic tagging engine.

Usage:
  engine.py [-D -d -m <str> -t <str> -f <str>]

Options:

  -m <str>, --method <str>                Method to call (smtag|tag|entity|panelize) [default: smtag]
  -t <str>, --text <str>                  Text input in unicode [default: Fluorescence microcopy images of GFP-Atg5 in fibroblasts from Creb1-/- mice after bafilomycin treatment.].
  -f <str>, --format <str>                Format of the output [default: xml]
  -D, --debug                             Debug mode to see the successive processing steps in the engine.
  -d, --demo                              Demo with a long sample.
"""	

#-g <str>, --tag <str>                   XML tag to update when using the role prediction method [default: sd-tag]

import re
from torch import nn
import torch
from docopt import docopt
from common.importexport import load_model
from common.utils import tokenize, timer
from train.builder import Concat
from common.converter import TString
from common.mapper import Catalogue
from predict.binarize import Binarized
from predict.predictor import SimplePredictor, ContextualPredictor
from predict.serializer import Serializer
from common.config import PROD_DIR
from common.viz import Show

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
        # get indices of output channels (of the source module) in the order require by input semantics (of the receiving module).
        # features should match by type or by role or by both?
        # my_index uses equal_type() to tests for equality of the type attribute of Concept objects
        matches = [concept.my_index(output_semantics) for concept in input_semantics] # finds the position of the required input concept in the list of output concepts
        matches = [m for m in matches if m is not None] # filter(None, matches) would filter out zeros which would be wrong; we could also use list(filter(lambda x: x or x==0, matches)) to filter None or [] or "" but not clearer
        self.indices = matches

    def forward(self, x):
        '''
        returns the tensor where second dimension (channels) of x are reordered to match the needs of the downstream B module.
        '''
        return x[ : , self.indices, : ]


class SmtagEngine:

    DEBUG = False

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
                    (load_model('exp_assay.zip', PROD_DIR), ''),
                    (load_model('disease.zip', PROD_DIR), '')
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

    def __panels(self, input_t_string):
        p = SimplePredictor(self.models['panelizer'])
        binarized = p.pred_binarized(input_t_string, self.models['panelizer'].output_semantics)
        return binarized

    def __entity(self, input_t_string):
        p = SimplePredictor(self.models['entity'])
        binarized = p.pred_binarized(input_t_string, self.models['entity'].output_semantics)
        return binarized

    def __reporter(self, input_t_string):
        p = SimplePredictor(self.models['only_once'])
        binarized = p.pred_binarized(input_t_string, self.models['only_once'].output_semantics)
        return binarized

    def __context(self, input_t_string, anonymization_marks):
        context_p = ContextualPredictor(self.models['context'])
        binarized = context_p.pred_binarized(input_t_string, anonymization_marks, self.models['context'].output_semantics)
        return binarized

    def __entity_and_role(self, input_t_string):
        binarized = self.__entity(input_t_string)
        binarized_reporter =self.__reporter(input_t_string)
        binarized.erase_(binarized_reporter)
        binarized.cat_(binarized_reporter)
        rewire = Connector(self.models['entity'].output_semantics, self.models['context'].anonymize_with)
        anonymization_marks = rewire.forward(binarized.marks)
        context_binarized = self.__context(input_t_string, anonymization_marks)
        binarized.cat_(context_binarized)
        return binarized

    def __all(self, input_string):

        input_t_string = TString(input_string)
        if self.DEBUG:
            print(input_t_string)

        #PREDICT PANELS
        binarized_panels = self.__panels(input_t_string)

        # PREDICT ENTITIES
        binarized_entities = self.__entity(input_t_string)
        if self.DEBUG:
            print("\n0: binarized.marks after entity ({})".format(" x ".join([str(s) for s in binarized_entities.marks.size()]))); Show.print_pretty(binarized_entities.marks)
            print("output semantics: ", "; ".join([str(e) for e in binarized_entities.output_semantics]))

        cumulated_output = binarized_entities.clone()
        cumulated_output.cat_(binarized_panels)
        if self.DEBUG:
            print("\n1: cumulated_output.marks after panel and entity ({})".format(" x ".join([str(s) for s in cumulated_output.marks.size()]))); Show.print_pretty(cumulated_output.marks)
            print("output semantics: ", "; ".join([str(e) for e in cumulated_output.output_semantics]))

        # PREDICT REPORTERS
        binarized_reporter = self.__reporter(input_t_string)
        if self.DEBUG:
            print("\n2: binarized_reporter.marks ({})".format(" x ".join([str(s) for s in binarized_reporter.marks.size()])));Show.print_pretty(binarized_reporter.marks)
            print("output semantics: ", "; ".join([str(e) for e in binarized_reporter.output_semantics]))

        binarized_entities.erase_(binarized_reporter) # will it erase itself? need to assume output is 1 channel only
        if self.DEBUG:
            print("\n2: binarized_reporter.marks ({})".format(" x ".join([str(s) for s in binarized_reporter.marks.size()])));Show.print_pretty(binarized_reporter.marks)
            print("\n3: binarized_entities.marks after erase_(reporter) ({})".format(" x ".join([str(s) for s in binarized_entities.marks.size()]))); Show.print_pretty(binarized_entities.marks)
            print("output semantics: ", "; ".join([str(e) for e in binarized_entities.output_semantics]))

        cumulated_output.cat_(binarized_reporter) # add reporter prediction to output features
        if self.DEBUG:
            print("\n4: cumulated_output.marks after cat_(reporter) ({})".format(" x ".join([str(s) for s in cumulated_output.marks.size()]))); Show.print_pretty(cumulated_output.marks)
            print("output semantics: ", "; ".join([str(e) for e in cumulated_output.output_semantics]))

        # select and match by type the predicted entity marks to be fed to the second context_p semantics from context model.
        rewire = Connector(self.models['entity'].output_semantics, self.models['context'].anonymize_with)
        anonymization_marks = rewire.forward(binarized_entities.marks) # should not include reporter marks; 
        if self.DEBUG:
            print("\n5: rewiring models['entity'].output_semantics and models['context'].anonymize_with", ", ".join([str(e) for e in self.models['entity'].output_semantics]), ", ".join([str(e) for e in self.models['context'].anonymize_with]))
            print("anonymization_marks ({})".format(" x ".join([str(s) for s in anonymization_marks.size()]))); Show.print_pretty(anonymization_marks)

        # PREDICT ROLES ON NON REPORTER ENTITIES
        context_binarized = self.__context(input_t_string, anonymization_marks)
        if self.DEBUG:
            print("\n6: context_binarized ({})".format(" x ".join([str(s) for s in context_binarized.marks.size()]))); Show.print_pretty(context_binarized.marks)

        #concatenate entity and output_semantics before calling Serializer()
        cumulated_output.cat_(context_binarized)
        if self.DEBUG:
            print("\n7: final cumulated_output.marks ({})".format(" x ".join([str(s) for s in cumulated_output.marks.size()])));Show.print_pretty(cumulated_output.marks)
            print("output semantics: ", "; ".join([str(e) for e in cumulated_output.output_semantics]))
        
        return cumulated_output

    def __serialize(self, output):
        ml = Serializer().serialize(output)
        return ml[0]

    @timer
    def entity(self, input_string):
        return self.__serialize(self.__entity(TString(input_string)))

    @timer
    def tag(self, input_string):
        return self.__serialize(self.__entity_and_role(TString(input_string)))

    @timer
    def smtag(self, input_string):
        return self.__serialize(self.__all(input_string))

    @timer
    def add_roles(self, input_xml):
        pass # need to implement an xml updater

    @timer
    def panelizer(self, input_string):
        return self.__serialize(self.__panels(TString(input_string)))

if __name__ == "__main__":
    arguments = docopt(__doc__, version='0.1')
    input_string = arguments['--text']
    method = arguments['--method']
    DEBUG = arguments['--debug']
    DEMO = arguments['--demo']
    if DEMO:
        input_string = '''The indicated panel of cell lines was exposed to either normoxia (20% O2) or hypoxia (1% O2) for up to 48 h prior to RNA and protein extraction.

A, B (A) LIMD1 mRNA and (B) protein levels were increased following hypoxic exposure.

C. Densitometric analysis of (B).

D. The LIMD1 promoter contains a hypoxic response element responsible for HIF binding and transcriptional activation of LIMD1. Three predicted HRE elements were individually deleted within the context of the wild‐type LIMD1 promoter‐driven Renilla luciferase.

E. Reporter constructs in (D) were expressed in U2OS cells and exposed to hypoxia for the indicated time‐points. Luciferase activity was then assayed and normalised to firefly control. Data are displayed normalised to the normoxic value for each construct. Deletion of the third HRE present within the LIMD1 promoter (ΔHRE3) inhibited hypoxic induction of LIMD1 transcription.

F, G (F) Sequence alignment and (G) sequence logo of LIMD1 promoters from the indicated species demonstrate that the HRE3 consensus sequence is highly conserved.'''

    input_string = re.sub("[\n\r\t]", " ", input_string)
    input_string = re.sub(" +", " ", input_string)
    
    engine = SmtagEngine()
    engine.DEBUG = DEBUG

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
