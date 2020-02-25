# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
import torch.nn
from torch.nn import functional as F
from math import ceil
from collections import namedtuple
from typing import List, Tuple
from ..common.converter import TString, StringList
from .decode import Decoder, CharLevelDecoder
from .markup import Serializer
from ..common.utils import tokenize, timer, Token
from ..common.embeddings import EMBEDDINGS
from ..common.mapper import Concept
from .. import config

# import cProfile
from time import time

PADDING_CHAR = config.padding_char

class Predictor: #(SmtagModel?) # eventually this should be fused with SmtagModel class and subclassed

    def __init__(self, model: 'CombinedModel', tag='sd-tag', format='xml'):
        self.model = model
        self.tag = tag
        self.format = format

    def padding(self, input_t_strings: TString) -> Tuple[TString, int]:
        # MOVE THIS TO ENGINE PREPROCESS
        # 0123456789012345678901234567890
        #        this cat               len(input)==8, min_size==10, min_padding=5
        #       +this cat+              pad to bring to min_size
        #  _____+this cat+_____         add min_padding on both sides
        min_size= config.min_size
        min_padding = config.min_padding
        padding_length = ceil(max(min_size - len(input_t_strings), 0) / 2) + min_padding
        pad = TString(StringList([PADDING_CHAR * padding_length] * input_t_strings.depth))
        # pad = padding_char_t.repeat(padding_length)
        padded_t_strings = pad + input_t_strings + pad
        return padded_t_strings, padding_length

    @staticmethod
    def embed(x: TString) -> torch.Tensor:
        if torch.cuda.is_available():
            return EMBEDDINGS(x.tensor.cuda())
        else:
            return EMBEDDINGS(x.tensor)

    def forward(self, input_t_strings:TString) -> torch.Tensor:
        # PADD TO MINIMAL LENGTH
        safely_padded, padding_length = self.padding(input_t_strings)
        
        # EMBEDDING
        x = self.embed(safely_padded)

        # PREDICTION
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(x) #.float() # prediction is 3D 1 x C x L
            prediction = torch.softmax(prediction) # to get 0..1 positive scores
            self.model.train()
        if torch.cuda.is_available():
            prediction = prediction.cpu()

        # RESTORE ORIGINAL LENGTH 
        prediction = prediction[ : , : , padding_length : len(safely_padded)-padding_length]
        return prediction

    def decode(self, input_strings: StringList, token_lists: List[List[Token]], prediction: torch.Tensor, semantic_groups: List[Concept]):
        decoded = Decoder(input_strings, prediction, self.model.semantic_groups)
        #######
        # cProfile.runctx('decoded.decode(token_lists)', None, locals())
        #######
        decoded.decode(token_lists)
        return decoded

    def predict(self, input_t_strings: TString, token_lists: List[List[Token]]) -> Decoder:
        prediction = self.forward(input_t_strings)
        decoded = self.decode(input_t_strings.toStringList(), token_lists, prediction, self.model.semantic_groups)
        return decoded

class ContextualPredictor(Predictor):

    def __init__(self, model: 'ContextCombinedModel', tag='sd-tag', format='xml') -> TString:
        super(ContextualPredictor, self).__init__(model, tag, format)

    @staticmethod
    def anonymize(for_anonymization: Decoder, group: str, concept_to_anonymize: Concept, mark_char: str = config.marking_char) -> TString:
        res_list = []
        for n in range(for_anonymization.N):
            concepts = for_anonymization.concepts[n][group]
            token_list = for_anonymization.token_lists[n]
            res = list(for_anonymization.input_strings[n]) # explode the string
            for token, concept in zip(token_list, concepts):
                if concept == concept_to_anonymize:
                    res[token.start:token.stop] = [mark_char] * token.length
            res = "".join(res) # reassemble the string
            res_list.append(res)
        return TString(StringList(res_list))

    def forward(self, input_t_strings_list: List[TString]):
        # PADDING AND EMBEDDING OF THE *LIST* OF TStrings THAT WERE EACH ANONMYMIZED IN A DIFFERENT WAY
        safely_padded = []
        for inp in input_t_strings_list:
            padded, padding_length = self.padding(inp)
            safely_padded.append(padded)

        if torch.cuda.is_available():
            x_list = [EMBEDDINGS(p.tensor.cuda()) for p in safely_padded]
        else:
            x_list = [EMBEDDINGS(p.tensor) for p in safely_padded]
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(x_list) # ContextCombinedModel takes List[torch.Tensor] as input and -> prediction is 3D N x C x L
            prediction = torch.softmax(prediction) # to get 0..1 positive scores
            self.model.train()
        if torch.cuda.is_available():
            prediction = prediction.cpu()

        # RESTORE ORIGINAL LENGTH 
        prediction = prediction[ : , : , padding_length : len(inp)+padding_length]
        return prediction

    def predict(self, for_anonymization: Decoder) -> Decoder:
        input_t_strings_list = []
        for anonymization in self.model.anonymize_with:
            group = anonymization['group']
            concept = anonymization['concept']
            anonymized_tstring = self.anonymize(for_anonymization, group, concept)
            input_t_strings_list.append(anonymized_tstring)
        
        prediction = self.forward(input_t_strings_list)
        input_strings = for_anonymization.input_strings
        token_lists = for_anonymization.token_lists
        decoded = self.decode(input_strings, token_lists, prediction, self.model.semantic_groups)
        return decoded


class CharLevelPredictor(Predictor):

    def decode(self, input_strings: StringList, token_lists: List[List[Token]], prediction: torch.Tensor, semantic_groups: List[Concept]):
        decoded = CharLevelDecoder(input_strings, prediction, self.model.semantic_groups)
        decoded.decode(token_lists)
        return decoded