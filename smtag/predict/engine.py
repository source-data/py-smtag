# -*- coding: utf-8 -*-
#T. Lemberger, 2018

"""
SmartTag semantic tagging engine.

Usage:
  engine.py [-D -d -m <str> -t <str> -f <str> -w <str> -g <str>]

Options:

  -m <str>, --method <str>                Method to call (smtag|tag|entity|role|panelize) [default: smtag]
  -t <str>, --text <str>                  Text input in unicode [default: Fluorescence microcopy images of GFP-Atg5 in fibroblasts from Creb1-/- mice after bafilomycin treatment.].
  -f <str>, --format <str>                Format of the output [default: xml]
  -w <str>, --working_directory <str>     Working directory where to read cartrigdes from (i.e. path where the `rack` folder is located)
  -D, --debug                             Debug mode to see the successive processing steps in the engine.
  -g <str>, --tag <str>                   XML tag to use when tagging concepts [default: sd-tag]
  -d, --demo                              Demo with a long sample.
"""



import re
from torch import nn
import torch
from docopt import docopt
from collections import OrderedDict
from xml.etree.ElementTree import tostring, fromstring, Element
from ..common.importexport import load_model
from ..common.utils import tokenize, timer
from ..train.builder import Concat
from ..common.converter import TString
from ..common.mapper import Catalogue
from ..datagen.encoder import XMLEncoder
from .decode import CharLevelDecoder, Decoder
from .predictor import Predictor, ContextualPredictor, CharLevelPredictor
from .markup import Serializer
from .updatexml import updatexml_
from .. import config
from ..common.viz import Show


# from test.test_max_window_unet import test_model

# maybe should be in builder.py
class CombinedModel(nn.Module):#SmtagModel?
    '''
    This module takes a list of SmtagModels and concatenates their output along the second dimension (feature axe).
    The output_semantics keeps track of the semantics of each module in the right order.
    '''

    def __init__(self, models: OrderedDict):
        super(CombinedModel, self).__init__()
        self.semantic_groups = OrderedDict()
        for group in models: # replace by self.module_list = nn.ModuleList(models) in torch 1.0.0
            model = models[group]
            name = 'unet2__'+ str(group)
            self.add_module(name, model) # self.module_list.append(model)
            self.semantic_groups[group] = model.output_semantics
        self.concat = Concat(1)

    def forward(self, x):
        y_list = []
        for n, m in self.named_children():
            if re.match('unet2__', n): # the child module is one of the unet models
                y_list.append(m(x))
        y = self.concat(y_list)
        return y

class ContextCombinedModel(CombinedModel):

    def __init__(self, models: OrderedDict):
        model_list = OrderedDict()
        self.anonymize_with = []
        for group in models:
            model, anonymization = models[group]
            self.anonymize_with.append(anonymization)
            model_list[group] = model
        super(ContextCombinedModel, self).__init__(model_list) # PROBABLY WRONG: each model needs to be run on different anonymization input

    def forward(self, x_list): # takes a list of inputs each specifically anonymized for each context model
        y_list = []
        for (n, m), x in zip(self.named_children(), x_list):
            if re.match('unet2__', n): # the child module is one of the unet models;
                y_list.append(m(x))
        y = self.concat(y_list)
        return y

class SmtagEngine:

    DEBUG = False

    def __init__(self):
        self.entity_models = CombinedModel(OrderedDict([
            ('entities', load_model(config.model_entity, config.prod_dir)),
            ('diseases', load_model(config.model_disease, config.prod_dir)),
            ('exp_assay', load_model(config.model_assay, config.prod_dir)) # can in fact be co-trained with entities since mutually exclusive
        ]))
        self.reporter_models = CombinedModel(OrderedDict([
            ('reporter', load_model(config.model_geneprod_reporter, config.prod_dir))
        ]))
        self.context_models = ContextCombinedModel(OrderedDict([
            ('geneprod_roles',
               (load_model(config.model_geneprod_role, config.prod_dir), {'group': 'entities', 'concept': Catalogue.GENEPROD})
            ),
            # ('small_molecule_role', 
            #     (load_model(config.model_molecule_role, config.prod_dir), {'group': 'entities', 'concept': Catalogue.SMALL_MOLECULE})
            # )
        ]))
        self.panelize_model = CombinedModel(OrderedDict([
            ('panels', load_model(config.model_panel_stop, config.prod_dir))
        ]))

    def __panels(self, input_t_string: TString, token_list) -> CharLevelDecoder:
        decoded = CharLevelPredictor(self.panelize_model).predict(input_t_string, token_list)
        return decoded

    def __entity(self, input_t_string: TString, token_list) -> Decoder:
        decoded = Predictor(self.entity_models).predict(input_t_string, token_list)
        return decoded

    def __reporter(self, input_t_string: TString, token_list) -> Decoder:
        decoded = Predictor(self.reporter_models).predict(input_t_string, token_list)
        return decoded

    def __context(self, entities: Decoder) -> Decoder: # entities carries the copy of the input_string and token_list
        decoded = ContextualPredictor(self.context_models).predict(entities)
        return decoded

    def __entity_and_role(self, input_t_string, token_list) -> Decoder:
        entities = self.__entity(input_t_string, token_list)
        output = entities.clone() # clone just in case, test if necessary...should not be
        reporter = self.__reporter(input_t_string, token_list)
        entities_less_reporter = entities.erase_with(reporter, ('reporter', Catalogue.REPORTER), ('entities', Catalogue.GENEPROD))
        output.cat_(reporter)
        context = self.__context(entities_less_reporter)
        output.cat_(context)
        return output

    def __role_from_pretagged(self, input_xml: Element) -> Decoder:
        input_string = ''.join([s for s in input_xml.itertext()])
        input_t_string = TString(input_string)
        token_list = tokenize(input_string)['token_list']
        encoded = XMLEncoder.encode(input_xml) # 2D tensor, single example
        encoded.unsqueeze_(0) # 3D
        semantic_groups = OrderedDict([('all_concepts', Catalogue.standard_channels)])
        entities = Decoder(input_string, encoded, semantic_groups)
        entities.decode(token_list)
        reporter = self.__reporter(input_t_string, token_list)
        entities_less_reporter = entities.erase_with(reporter, ('reporter', Catalogue.REPORTER), ('entities', Catalogue.GENEPROD))
        output = reporter # there was a clone() here??
        context = self.__context(entities_less_reporter)
        output.cat_(context)
        return output

    def __all(self, input_t_string, token_list):

        if self.DEBUG:
            show = Show()
            print("\nText:")
            print("    "+str(input_t_string))

        #PREDICT PANELS
        panels = self.__panels(input_t_string, token_list)
        output = panels
        if self.DEBUG:
            B, C, L = output.prediction.size()
            print(f"\n1: after panels: {output.semantic_groups} {B}x{C}x{L}")
            print(show.print_pretty(output.prediction))

        # PREDICT ENTITIES
        entities = self.__entity(input_t_string, token_list)
        if self.DEBUG:
            B, C, L = entities.prediction.size()
            print(f"\n1: after entity: {entities.semantic_groups} {B}x{C}x{L}")
            print(show.print_pretty(entities.prediction))
        output.cat_(entities.clone()) 

        # PREDICT REPORTERS
        reporter = self.__reporter(input_t_string, token_list)
        if self.DEBUG:
            B, C, L = entities.prediction.size()
            print(f"\n2: after reporter: {reporter.semantic_groups} {B}x{C}x{L}")
            print(show.print_pretty(reporter.prediction))
        output.cat_(reporter) # add reporter prediction to output features
        entities_less_reporter = entities.erase_with(reporter, ('reporter', Catalogue.REPORTER), ('entities', Catalogue.GENEPROD)) # how ugly!
        if self.DEBUG:
            B, C, L = entities_less_reporter.prediction.size()
            print(f"\n3: after entity.erase_(reporter): {entities_less_reporter.semantic_groups} {B}x{C}x{L}")
            print(show.print_pretty(entities_less_reporter.prediction))

        # PREDICT ROLES ON NON REPORTER ENTITIES
        context = self.__context(entities_less_reporter)
        if self.DEBUG:
            B, C, L = context.prediction.size()
            print(f"\n4: after context: {context.semantic_groups} {B}x{C}x{L}")
            print(show.print_pretty(context.prediction))
        output.cat_(context)

        if self.DEBUG:
            B, C, L = output.prediction.size()
            print(f"\n5: concatenated output: {output.semantic_groups} {B}x{C}x{L}")
            print(show.print_pretty(output.prediction))

        return output

    def __serialize(self, output, sdtag="sd-tag", format="xml"):
        output.fuse_adjascent()
        ml = Serializer(tag=sdtag, format=format).serialize(output)
        return ml # engine works with single example

    def __string_preprocess(self, input_string):
        return TString(input_string), tokenize(input_string)['token_list']

    @timer
    def entity(self, input_string, sdtag, format):
        input_t_string, token_list = self.__string_preprocess(input_string)
        return self.__serialize(self.__entity(input_t_string, token_list), sdtag, format)

    @timer
    def tag(self, input_string, sdtag, format):
        input_t_string, token_list = self.__string_preprocess(input_string)
        return self.__serialize(self.__entity_and_role(input_t_string, token_list), sdtag, format)

    @timer
    def smtag(self, input_string, sdtag, format):
        input_t_string, token_list = self.__string_preprocess(input_string)
        return self.__serialize(self.__all(input_t_string, token_list), sdtag, format)

    @timer
    def role(self, input_xml_string:str, sdtag):
        input_xml = fromstring(input_xml_string)
        updatexml_(input_xml, self.__role_from_pretagged(input_xml), pretag=fromstring('<'+sdtag+'/>'))
        return tostring(input_xml)

    @timer
    def panelizer(self, input_string, format):
        input_t_string, token_list = self.__string_preprocess(input_string)
        return self.__serialize(self.__panels(input_t_string, token_list), format=format)

def main():
    arguments = docopt(__doc__, version='0.1')
    input_string = arguments['--text']
    method = arguments['--method']
    DEBUG = arguments['--debug']
    DEMO = arguments['--demo']
    sdtag = arguments['--tag']
    format = arguments['--format']
    if arguments['--working_directory']:
        config.working_directory = arguments['--working_directory']
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
        print(engine.smtag(input_string, sdtag, format))
    elif method == 'panelize':
        print(engine.panelizer(input_string, format))
    elif method == 'tag':
        print(engine.tag(input_string, sdtag, format))
    elif method == 'entity':
        print(engine.entity(input_string, sdtag, format))
    elif method == 'role':
        print(engine.role(input_string, sdtag)) # can only be xml format
    else:
        print("unknown method {}".format(method))

if __name__ == "__main__":
    main()
