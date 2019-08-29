# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import re
from torch import nn
import torch
from docopt import docopt
from collections import OrderedDict
from xml.etree.ElementTree import tostring, fromstring, Element
from ..common.utils import tokenize, timer
from ..common.converter import TString
from ..common.mapper import Catalogue
from ..common.importexport import load_model
from ..datagen.encoder import XMLEncoder
from ..datagen.context import VisualContext
from .decode import CharLevelDecoder, Decoder
from .predictor import Predictor, ContextualPredictor, CharLevelPredictor
from .markup import Serializer
from .updatexml import updatexml_
from ..common.viz import Show
from .. import config


class CombinedModel(nn.Module):
    '''
    This module takes a list of SmtagModels and concatenates their output along the second dimension (feature axe).
    The output_semantics keeps track of the semantics of each module in the right order.
    '''

    def __init__(self, models: OrderedDict):
        super(CombinedModel, self).__init__()
        self.semantic_groups = OrderedDict()
        self.model_list = nn.ModuleDict(models)
        self.semantic_groups = {g:models[g].output_semantics for g in models}

    def forward(self, x, viz_context):
        y_list = []
        for group in self.semantic_groups:
            model = self.model_list[group]
            y_list.append(model(x, viz_context))
        y = torch.cat(y_list, 1)        
        return y

class ContextCombinedModel(nn.Module):

    def __init__(self, models: OrderedDict):
        super(ContextCombinedModel, self).__init__()
        self.semantic_groups = OrderedDict()
        self.anonymize_with = []
        self.model_list = nn.ModuleList()
        for group in models:
            model, anonymization = models[group]
            self.anonymize_with.append(anonymization)
            self.model_list.append(model)
            self.semantic_groups[group] = model.output_semantics
         #super(ContextCombinedModel, self).__init__(self.model_list) # PROBABLY WRONG: each model needs to be run on different anonymization input

    def forward(self, x_list, viz_context): # takes a list of inputs each specifically anonymized for each context model
        y_list = []
        for m, x in zip(self.model_list, x_list):
            y_list.append(m(x, viz_context))
        y = torch.cat(y_list, 1)
        return y

class Cartridge():

    def __init__(self, entity_models: CombinedModel, reporter_models: CombinedModel, context_models:CombinedModel, panelize_model:CombinedModel, viz_preprocessor:VisualContext):
        self.entity_models = entity_models
        self.reporter_models = reporter_models
        self.context_models = context_models
        self.panelize_model = panelize_model
        self.viz_preprocessor = viz_preprocessor


class SmtagEngine:

    DEBUG = False

    def __init__(self, cartridge:Cartridge):
        self.entity_models = cartridge.entity_models
        self.reporter_models = cartridge.reporter_models
        self.context_models = cartridge.context_models
        self.panelize_model = cartridge.panelize_model
        self.viz_context_processor = cartridge.viz_preprocessor

    def __panels(self, input_t_string: TString, token_list, viz_context) -> CharLevelDecoder:
        decoded = CharLevelPredictor(self.panelize_model).predict(input_t_string, token_list, viz_context)
        if self.DEBUG:
            B, C, L = decoded.prediction.size()
            print(f"\nafter panels: {decoded.semantic_groups} {B}x{C}x{L}")
            print(Show().print_pretty(decoded.prediction))
        return decoded

    def __entity(self, input_t_string: TString, token_list, viz_context) -> Decoder:
        decoded = Predictor(self.entity_models).predict(input_t_string, token_list, viz_context)
        if self.DEBUG:
            B, C, L = decoded.prediction.size()
            print(f"\nafter entity: {decoded.semantic_groups} {B}x{C}x{L}")
            print(Show().print_pretty(decoded.prediction))
        return decoded

    def __reporter(self, input_t_string: TString, token_list, viz_context) -> Decoder:
        decoded = Predictor(self.reporter_models).predict(input_t_string, token_list, viz_context)
        if self.DEBUG:
            B, C, L = decoded.prediction.size()
            print(f"\nafter reporter: {decoded.semantic_groups} {B}x{C}x{L}")
            print(Show().print_pretty(decoded.prediction))
        return decoded

    def __context(self, entities: Decoder, viz_context) -> Decoder: # entities carries the copy of the input_string and token_list
        decoded = ContextualPredictor(self.context_models).predict(entities, viz_context)
        if self.DEBUG:
            B, C, L = decoded.prediction.size()
            print(f"\nafter context: {decoded.semantic_groups} {B}x{C}x{L}")
            print(Show().print_pretty(decoded.prediction))
        return decoded

    def __entity_and_role(self, input_t_string, token_list, viz_context) -> Decoder:
        entities = self.__entity(input_t_string, token_list, viz_context)
        output = entities.clone() # clone just in case, test if necessary...should not be
        reporter = self.__reporter(input_t_string, token_list, viz_context)
        entities_less_reporter = entities.erase_with(reporter, ('reporter', Catalogue.REPORTER), ('entities', Catalogue.GENEPROD))
        output.cat_(reporter)
        context = self.__context(entities_less_reporter, viz_context)
        output.cat_(context)

        return output

    def __role_from_pretagged(self, input_xml: Element, viz_context) -> Decoder:
        input_string = ''.join([s for s in input_xml.itertext()])
        input_t_string = TString(input_string)
        token_list = tokenize(input_string)['token_list']
        encoded = XMLEncoder.encode(input_xml) # 2D tensor, single example
        encoded.unsqueeze_(0) # 3D byteTensor!
        semantic_groups = OrderedDict([('entities', Catalogue.standard_channels)])
        entities = Decoder(input_string, encoded.float(), semantic_groups)
        entities.decode(token_list)
        reporter = self.__reporter(input_t_string, token_list, viz_context)
        entities_less_reporter = entities.erase_with(reporter, ('reporter', Catalogue.REPORTER), ('entities', Catalogue.GENEPROD))
        output = reporter # there was a clone() here??
        context = self.__context(entities_less_reporter, viz_context)
        output.cat_(context)
        return output

    def __all(self, input_t_string, token_list, viz_context):

        if self.DEBUG:
            print("\nText:")
            print("    "+str(input_t_string))

        panels = self.__panels(input_t_string, token_list, viz_context)
        output = panels

        entities = self.__entity(input_t_string, token_list, viz_context)
        output.cat_(entities.clone())

        reporter = self.__reporter(input_t_string, token_list, viz_context)
        output.cat_(reporter) # add reporter prediction to output features

        entities_less_reporter = entities.erase_with(reporter, ('reporter', Catalogue.REPORTER), ('entities', Catalogue.GENEPROD)) # how ugly!
        context = self.__context(entities_less_reporter, viz_context)
        output.cat_(context)

        if self.DEBUG:
            B, C, L = output.prediction.size()
            print(f"\nfinal concatenated output: {output.semantic_groups} {B}x{C}x{L}")
            print(Show().print_pretty(output.prediction))

        return output

    def __serialize(self, output, sdtag="sd-tag", format="xml"):
        output.fuse_adjascent()
        ml = Serializer(tag=sdtag, format=format).serialize(output)
        return ml # engine works with single example

    def __string_preprocess(self, input_string):
        return TString(input_string), tokenize(input_string)['token_list']

    def __img_preprocess(self, img):
        # what does get_context return if img is None? or torch.Tensor(0)?
        # context returns a black image if img is None
        # the Context module returns an empty list if context is torch.Tensor(0)
        # what to do if no img for _with_viz model
        # what to do if img for _no_viz
        if img is not None:
            viz_context = self.viz_context_processor.get_context(img)
            vectorized = viz_context.view(1, -1) # WATCH OUT: torch.Tensor(0).view(1,-1).size() is torch.Size([1, 0])
        else:
            vectorized = torch.Tensor(0) # the model will then skip the Context module and only uses the unet with the text
        return vectorized
    
    @timer
    def entity(self, input_string, img, sdtag, format):
        input_t_string, token_list = self.__string_preprocess(input_string)
        viz_context = self.__img_preprocess(img)
        pred = self.__entity(input_t_string, token_list, viz_context)
        return self.__serialize(pred, sdtag, format)

    @timer
    def tag(self, input_string, img, sdtag, format):
        input_t_string, token_list = self.__string_preprocess(input_string)
        viz_context = self.__img_preprocess(img)
        pred = self.__entity_and_role(input_t_string, token_list, viz_context)
        return self.__serialize(pred, sdtag, format)

    @timer
    def smtag(self, input_string, img, sdtag, format):
        input_t_string, token_list = self.__string_preprocess(input_string)
        viz_context = self.__img_preprocess(img)
        pred = self.__all(input_t_string, token_list, viz_context)
        return self.__serialize(pred, sdtag, format)

    @timer
    def role(self, input_xml_string:str, img, sdtag):
        input_xml = fromstring(input_xml_string)
        viz_context = self.__img_preprocess(img)
        pred = self.__role_from_pretagged(input_xml, viz_context)
        updatexml_(input_xml, pred, pretag=fromstring('<'+sdtag+'/>'))
        return tostring(input_xml) # tostring() returns bytes...

    @timer
    def panelizer(self, input_string, img, format):
        input_t_string, token_list = self.__string_preprocess(input_string)
        viz_context = self.__img_preprocess(img) # DEBUG
        pred = self.__panels(input_t_string, token_list, viz_context)
        return self.__serialize(pred, format=format)

def main():
    parser = config.create_argument_parser_with_defaults(description='SmartTag semantic tagging engine.')
    parser.add_argument('-m', '--method', default='smtag', help='Method to call (smtag|tag|entity|role|panelize)')
    parser.add_argument('-t', '--text', default='', help='Text input in unicode')
    parser.add_argument('-f', '--format', default='xml', help='Format of the output')
    parser.add_argument('-D', '--debug', action='store_true', help='Debug mode to see the successive processing steps in the engine.')
    parser.add_argument('-g', '--tag', default='sd-tag', help='XML tag to use when tagging concepts')
    parser.add_argument('-d', '--demo', action='store_true', help='Demo with a long sample')

    arguments = parser.parse_args()
    input_string = arguments.text
    method = arguments.method
    DEBUG = arguments.debug
    DEMO = arguments.demo
    sdtag = arguments.tag
    format = arguments.format
    from .cartridges import NO_VIZ
    
    if DEMO:
        input_string = '''The indicated panel of cell lines was exposed to either normoxia (20% O2) or hypoxia (1% O2) for up to 48 h prior to RNA and protein extraction.

(A and B) (A) LIMD1 mRNA and (B) protein levels were increased following hypoxic exposure.

(C) Densitometric analysis of (B).

(D) The LIMD1 promoter contains a hypoxic response element responsible for HIF binding and transcriptional activation of LIMD1. Three predicted HRE elements were individually deleted within the context of the wild‐type LIMD1 promoter‐driven Renilla luciferase.

(E) Reporter constructs in (D) were expressed in U2OS cells and exposed to hypoxia for the indicated time‐points. Luciferase activity was then assayed and normalised to firefly control. Data are displayed normalised to the normoxic value for each construct. Deletion of the third HRE present within the LIMD1 promoter (ΔHRE3) inhibited hypoxic induction of LIMD1 transcription.

(F) Sequence alignment and (G) sequence logo of LIMD1 promoters from the indicated species demonstrate that the HRE3 consensus sequence is highly conserved.'''

    input_string = re.sub("[\n\r\t]", " ", input_string)
    input_string = re.sub(" +", " ", input_string)
    # cv_img = torch.zeros(500, 500, 3).numpy()
    cv_img = None
    engine = SmtagEngine(NO_VIZ)
    engine.DEBUG = DEBUG
    
    if method == 'smtag':
        print(engine.smtag(input_string, cv_img, sdtag, format))
    elif method == 'panelize':
        print(engine.panelizer(input_string, cv_img, format))
    elif method == 'tag':
        print(engine.tag(input_string, cv_img, sdtag, format))
    elif method == 'entity':
        print(engine.entity(input_string, cv_img, sdtag, format))
    elif method == 'role':
        print(engine.role(input_string, cv_img, sdtag)) # can only be xml format
    else:
        print("unknown method {}".format(method))

if __name__ == "__main__":
    main()
