# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import re
from torch import nn
import torch
from docopt import docopt
from collections import OrderedDict
from typing import List, Tuple
from xml.etree.ElementTree import tostring, fromstring, Element
from ..common.utils import tokenize, Token, timer
from ..common.converter import TString, StringList
from ..common.mapper import Catalogue
from ..datagen.encoder import XMLEncoder
from .decode import CharLevelDecoder, Decoder
from .predictor import Predictor, ContextualPredictor, CharLevelPredictor
from .markup import Serializer
from .cartridges import Cartridge, NO_VIZ
from .updatexml import updatexml_list
from ..common.viz import Show
from .. import config


class SmtagEngine:

    DEBUG = False

    def __init__(self, cartridge: Cartridge):
        self.entity_models = cartridge.entity_models
        self.reporter_models = cartridge.reporter_models
        self.context_models = cartridge.context_models
        self.panelize_model = cartridge.panelize_model
        self.viz_context_processor = cartridge.viz_preprocessor

    @timer
    def __panels(self, input_t_strings: TString, token_lists: List[List[Token]]) -> CharLevelDecoder:
        decoded = CharLevelPredictor(self.panelize_model).predict(input_t_strings, token_lists)
        if self.DEBUG:
            B, C, L = decoded.prediction.size()
            print(f"\nafter panels: {decoded.semantic_groups} {B}x{C}x{L}")
            print(Show().print_pretty(decoded.prediction))
        return decoded

    @timer
    def __entity(self, input_t_strings: TString, token_lists: List[List[Token]]) -> Decoder:
        decoded = Predictor(self.entity_models).predict(input_t_strings, token_lists)
        if self.DEBUG:
            B, C, L = decoded.prediction.size()
            print(f"\nafter entity: {decoded.semantic_groups} {B}x{C}x{L}")
            print(Show().print_pretty(decoded.prediction))
        return decoded
    @timer
    def __reporter(self, input_t_strings: TString, token_lists: List[List[Token]]) -> Decoder:
        decoded = Predictor(self.reporter_models).predict(input_t_strings, token_lists)
        if self.DEBUG:
            B, C, L = decoded.prediction.size()
            print(f"\nafter reporter: {decoded.semantic_groups} {B}x{C}x{L}")
            print(Show().print_pretty(decoded.prediction))
        return decoded

    @timer
    def __context(self, entities: Decoder) -> Decoder: # entities carries the copy of the input_string and token_list
        decoded = ContextualPredictor(self.context_models).predict(entities)
        if self.DEBUG:
            B, C, L = decoded.prediction.size()
            print(f"\nafter context: {decoded.semantic_groups} {B}x{C}x{L}")
            print(Show().print_pretty(decoded.prediction))
        return decoded

    def __entity_and_role(self, input_t_strings: TString, token_lists: List[List[Token]]) -> Decoder:
        output = self.__entity(input_t_strings, token_lists)
        # output = entities.clone() # clone just in case, test if necessary...should not be
        reporter = self.__reporter(input_t_strings, token_lists)
        entities_less_reporter = output.erase_with(reporter, ('reporter', Catalogue.REPORTER), ('entities', Catalogue.GENEPROD))
        output.cat_(reporter)
        context = self.__context(entities_less_reporter)
        output.cat_(context)
        return output

    def __role_from_pretagged(self, input_xml_list: List[Element]) -> Decoder:
        input_strings = []
        for xml_str in input_xml_list:
            input_strings.append(''.join([s for s in xml_str.itertext()]))
        input_t_strings = TString(StringList(input_strings))
        token_lists = [tokenize(s)['token_list'] for s in input_strings]
        encoded_list = []
        for x in input_xml_list:
            encoded = XMLEncoder.encode(x) # 2D tensor, single example
            encoded.unsqueeze_(0) # 3D byteTensor!
            encoded_list.append(encoded)
        encoded_cat = torch.cat(encoded_list, 0)
        # add a feature for untagged characters; necessary for softmax classification
        no_tag_feature = encoded_cat.sum(1) # 3D 1 x C x L, is superposition of all features so far
        no_tag_feature.unsqueeze_(0)
        no_tag_feature = 1 - no_tag_feature # sets to 1 for char not tagged and to 0 for tagged characters
        B = encoded_cat.size(0)
        no_tag_feature = no_tag_feature.repeat([B, 1, 1])
        encoded_cat = torch.cat((encoded_cat, no_tag_feature.byte()), 1)
        semantic_groups = OrderedDict([('entities', Catalogue.standard_channels)])
        semantic_groups['entities'].append(Catalogue.UNTAGGED)
        entities = Decoder(StringList(input_strings), encoded_cat.float(), semantic_groups)
        entities.decode(token_lists)
        reporter = self.__reporter(input_t_strings, token_lists)
        entities_less_reporter = entities.erase_with(reporter, ('reporter', Catalogue.REPORTER), ('entities', Catalogue.GENEPROD))
        output = reporter # there was a clone() here??
        context = self.__context(entities_less_reporter)
        output.cat_(context)
        return output

    def __all(self, input_t_strings: TString, token_lists: List[List[Token]]):
        if self.DEBUG:
            print("\nText:")
            print("    "+str(input_t_strings))

        panels = self.__panels(input_t_strings, token_lists)
        output = panels

        entities = self.__entity(input_t_strings, token_lists)
        output.cat_(entities.clone())

        reporter = self.__reporter(input_t_strings, token_lists)
        output.cat_(reporter) # add reporter prediction to output features

        entities_less_reporter = entities.erase_with(reporter, ('reporter', Catalogue.REPORTER), ('entities', Catalogue.GENEPROD)) # how ugly!
        context = self.__context(entities_less_reporter)
        output.cat_(context)

        if self.DEBUG:
            B, C, L = output.prediction.size()
            print(f"\nfinal concatenated output: {output.semantic_groups} {B}x{C}x{L}")
            print(Show().print_pretty(output.prediction))

        return output

    @timer
    def __serialize(self, output: Decoder, sdtag="sd-tag", format="xml") -> List[str]:
        output.fuse_adjacent()
        ml = Serializer(tag=sdtag, format=format).serialize(output)
        return ml

    def __string_preprocess(self, input_strings: List[str]) -> Tuple[TString, List[List[Token]]]:
        if isinstance(input_strings, str):
            input_strings = [input_strings] # for backward compatibility
        input_t_strings = TString(StringList(input_strings)) # StringList makes sure all strings are of same length before stacking them into tensor format
        token_lists = [tokenize(s)['token_list'] for s in input_strings]
        return input_t_strings, token_lists

    @timer
    def __preprocess(self, input_strings: List[str]) -> Tuple[TString, List[List[Token]], List[torch.Tensor]]:
        input_t_strings, token_lists = self.__string_preprocess(input_strings)
        return input_t_strings, token_lists

    @timer
    def entity(self, input_strings: List[str], sdtag, format) -> List[str]:
        prepro = self.__preprocess(input_strings) # input_t_strings, token_lists, viz_contexts
        pred = self.__entity(*prepro)
        return self.__serialize(pred, sdtag, format)

    @timer
    def tag(self, input_strings: List[str], sdtag, format) -> List[str]:
        prepro = self.__preprocess(input_strings)
        pred = self.__entity_and_role(*prepro)
        return self.__serialize(pred, sdtag, format)

    @timer
    def smtag(self, input_strings: List[str], sdtag, format) -> List[str]:
        prepro = self.__preprocess(input_strings)
        pred = self.__all(*prepro)
        return self.__serialize(pred, sdtag, format)

    @timer
    def role(self, input_xml_strings: List[str], sdtag)  -> List[bytes]:
        input_xmls = [fromstring(s) for s in input_xml_strings]
        pred = self.__role_from_pretagged(input_xmls)
        updated_xml = updatexml_list(input_xmls, pred, pretag=fromstring('<'+sdtag+'/>')) # in place, xml Elements and List mutable
        updated_xml_bytes = [tostring(x) for x in updated_xml] # tostring() returns bytes...
        return updated_xml_bytes

    @timer
    def panelizer(self, input_strings: List[str], sdtag, format) -> List[str]:
        prepro = self.__preprocess(input_strings)
        pred = self.__panels(*prepro)
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
    engine = SmtagEngine(NO_VIZ)
    engine.DEBUG = DEBUG
    
    if method == 'smtag':
        print(engine.smtag([input_string], sdtag, format))
    elif method == 'panelize':
        print(engine.panelizer([input_string], sdtag, format))
    elif method == 'tag':
        print(engine.tag([input_string], sdtag, format))
    elif method == 'entity':
        print(engine.entity([input_string, input_string], sdtag, format))
    elif method == 'role':
        print(engine.role([input_string], sdtag)) # can only be xml format
    else:
        print("unknown method {}".format(method))

if __name__ == "__main__":
    main()
