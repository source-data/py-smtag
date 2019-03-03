# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
import torch.nn
from torch.nn import functional as F
from math import ceil
from collections import namedtuple
from ..common.converter import TString
from .decode import Decoder, CharLevelDecoder
from .markup import Serializer
from ..common.utils import tokenize, timer
from .. import config


PADDING_CHAR = config.padding_char
PADDING_CHAR_T = TString(config.padding_char)

class Predictor: #(SmtagModel?) # eventually this should be fused with SmtagModel class and subclassed

    def __init__(self, model: 'CombinedModel', tag='sd-tag', format='xml'):
        self.model = model
        self.tag = tag
        self.format = format

    def padding(self, input):
        # 0123456789012345678901234567890
        #        this cat               len(input)==8, min_size==10, min_padding=5
        #       +this cat+              pad to bring to min_size
        #  _____+this cat+_____         add min_padding on both sides
        min_size= config.min_size
        min_padding = config.min_padding
        padding_length = ceil(max(min_size - len(input), 0)/2) + min_padding
        pad = PADDING_CHAR_T.repeat(padding_length)
        padded_string = pad + input + pad
        return padded_string

    def forward(self, input):
        if isinstance(input, list):
            padded = [self.padding(inp) for inp in input]
            L = len(padded[0])
            padding_length = int((L - len(input[0])) / 2)
            x = [p.toTensor() for p in padded]
        else:
            padded = self.padding(input)
            L = len(padded)
            padding_length = int((L - len(input)) / 2)
            x = padded.toTensor()

        #PREDICTION
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(x) #.float() # prediction is 3D 1 x C x L
            prediction = torch.sigmoid(prediction) # to get 0..1 positive scores
            self.model.train()

        #remove safety padding
        prediction = prediction[ : , : , padding_length : L-padding_length]
        return prediction

    def decode(self, input_str, token_list, prediction, semantic_groups):
        decoded = Decoder(input_str, prediction, self.model.semantic_groups)
        decoded.decode(token_list)
        return decoded
    
    def predict(self, input_t_string, token_list):
        prediction = self.forward(input_t_string)
        decoded = self.decode(str(input_t_string), token_list, prediction, self.model.semantic_groups)
        return decoded

class ContextualPredictor(Predictor):

    def __init__(self, model: 'ContextCombinedModel', tag='sd-tag', format='xml') -> TString:
        super(ContextualPredictor, self).__init__(model, tag, format)

    @staticmethod
    def anonymize(for_anonymization, group, concept_to_anonymize, mark_char = config.marking_char):
        concepts = for_anonymization.concepts[group]
        token_list = for_anonymization.token_list
        res = list(for_anonymization.input_string)
        for token, concept in zip(token_list, concepts):
            if type(concept) == type(concept_to_anonymize):
                res[token.start:token.stop] = [mark_char] * token.length
        res = "".join(res)
        return TString(res)

    def predict(self, for_anonymization: Decoder) -> Decoder:
        prediction = []
        anonymized_t = []
        for anonymization in self.model.anonymize_with:
            group = anonymization['group']
            concept = anonymization['concept']
            anonymized_t.append(self.anonymize(for_anonymization, group, concept))
        prediction = self.forward(anonymized_t) # ContextCombinedModel takes list of anonymized inputs; ouch need to be all padded
        input_string = for_anonymization.input_string
        token_list = for_anonymization.token_list
        decoded = self.decode(input_string, token_list, prediction, self.model.semantic_groups) # input_string will be tokenized again; a waste, but maybe not worth the complication; could have an *args or somethign
        return decoded
        

class CharLevelPredictor(Predictor):

    def decode(self, input_str, token_list, prediction, semantic_groups):
        decoded = CharLevelDecoder(input_str, prediction, self.model.semantic_groups)
        decoded.decode(token_list)
        return decoded