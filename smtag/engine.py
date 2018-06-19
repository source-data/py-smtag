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
from smtag.predictor import EntityPredictor, SemanticsFromContextPredictor
from smtag.serializer import Serializer
from smtag.config import PROD_DIR
from smtag.viz import Show

# maybe should be in builer.py
class Combine(nn.Module):#SmtagModel?

    def __init__(self, model_list):
        super(Combine, self).__init__()
        self.model_list = []
        self.output_semantics = []
        self.anonymize_with = []
        for model, anonymize_with in model_list:
            name = 'unet2__'+'_'.join(model.output_semantics)
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

    def __init__(self, cartridge={}, dir=PROD_DIR):
        #change this to accept a 'cartridge' that descibes which models to load and that can be save to disk
        if cartridge:
            self.cartridge = cartridge
        else:
           self.cartridge = {
               'entities': [
                   {'model': 'geneprod.zip', 'anonymize_with': ''},
                   {'model': 'small_molecule.zip', 'anonymize_with': ''}
               ],
               'only_once': [
                   {'model': 'reporter_geneprod.zip', 'anonymize_with': ''}
               ],
               'context': [
                   {'model': 'causality_entities.zip', 'anonymize_with': 'geneprod'},
                   {'model': 'causality_small_mol.zip', 'anonymize_with': 'small_mol'}
               ]
           }

        entity_model_list = [
            (load_model('geneprod.zip', dir), [])
        ]
        self.entity_model = Combine(entity_model_list)

        reporter_model_list = [
            (load_model('reporter_geneprod.zip', dir), [])
        ]
        self.reporter_model = Combine(reporter_model_list)

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
        
        entity_p = EntityPredictor(self.entity_model)
        binarized = entity_p.pred_binarized(input_string, self.entity_model.output_semantics)

        # select and order the predicted marks to be fed to the second context_p semantics from context model.
        rewire = Connector(self.entity_model.output_semantics, self.context_model.anonymize_with)
        marks = rewire.forward(binarized.marks)

        context_p = SemanticsFromContextPredictor(self.context_model)
        context_binarized = context_p.pred_binarized(input_string, marks, self.context_model.output_semantics)
        
        #concatenate entity and output_semantics before calling Serializer()
        binarized.cat_(context_binarized)
        ml = Serializer().serialize(binarized)
        return ml[0]

    def entity_reporter_context(self, input_string):
        
        # PREDICT ENTITIES
        entity_p = EntityPredictor(self.entity_model)
        binarized_entities = entity_p.pred_binarized(input_string, self.entity_model.output_semantics)
        print("0: binarized.marks after entity"); Show.print_pretty(binarized_entities.marks)
        print("output semantics: ", "; ".join(binarized_entities.output_semantics))

        cumulated_output = binarized_entities.clone()
        print("1: cumulated_output.marks after entity"); Show.print_pretty(cumulated_output.marks)
        print("output semantics: ", "; ".join(cumulated_output.output_semantics))

        # PREDICT REPORTERS
        reporter_p = EntityPredictor(self.reporter_model)
        binarized_reporter = reporter_p.pred_binarized(input_string, self.reporter_model.output_semantics)
        print("2: binarized_reporter.marks");Show.print_pretty(binarized_reporter.marks)
        print("output semantics: ", "; ".join(binarized_reporter.output_semantics))

        # DANGER: NEED A MECHANISM TO MAP TAG-ONCE CHANNELS TO CHANNELS TO ERASE... ie a geneprod reporter should erase geneprod marks and not small mol
        # looks like output semantics is too simple: need separation of entity type and role and category... :-(
        # simple solution: as many reporter model should be loaded as entities are predicted.
        binarized_entities.erase_(binarized_reporter) # will it erase itself? need to assume output is 1 channel only
        # select and order the predicted entity marks to be fed to the second context_p semantics from context model.
        print("3: binarized_entities.marks after erase_(reporter)"); Show.print_pretty(binarized_entities.marks)
        print("output semantics: ", "; ".join(binarized_entities.output_semantics))

        cumulated_output.cat_(binarized_reporter) # add reporter prediction to output features
        # reporters are marked as such but need to be erased as entity to avoid being anonymized or included in subsequent predictions (only_once method?)
        print("4: cumulated_output.marks after cat_(reporter)"); Show.print_pretty(cumulated_output.marks)
        print("output semantics: ", "; ".join(cumulated_output.output_semantics))

        rewire = Connector(self.entity_model.output_semantics, self.context_model.anonymize_with)
        marks = rewire.forward(binarized_entities.marks) # should not include reporter marks; need to test
        print("5: marks"); Show.print_pretty(marks)

        # PREDICT ROLES ON NON REPORTER ENTITIES
        context_p = SemanticsFromContextPredictor(self.context_model)
        context_binarized = context_p.pred_binarized(input_string, marks, self.context_model.output_semantics)
        print("6: context_binarized"); Show.print_pretty(context_binarized.marks)

        #concatenate entity and output_semantics before calling Serializer()
        cumulated_output.cat_(context_binarized)
        print("7: final cumulated_output.marks");Show.print_pretty(cumulated_output.marks)
        print("output semantics: ", "; ".join(cumulated_output.output_semantics))
        
        ml = Serializer().serialize(cumulated_output)
        return ml[0]
    
    def add_roles(self, input_xml):
        pass

    def panels(self, input_string):
        pass
    
    def smtag(self, input_string):
        pass

if __name__ == "__main__":
    # PARSE ARGUMENTS
    arguments = docopt(__doc__, version='0.1')
    input_string = arguments['--text']

    print(SmtagEngine().entities(input_string))
