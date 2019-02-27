# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
import re
from typing import List, Tuple
from collections import OrderedDict
from copy import copy, deepcopy
from ..common.utils import xml_escape, timer, tokenize, Token
from ..common.mapper import Catalogue
from .. import config

FUSION_THRESHOLD = config.fusion_threshold

Tensor3D = torch.Tensor
Tensor2D = torch.Tensor

class Decoder:
    '''
    Generates feature codes for each semantic group from prediction tensor. 
    prediction can be the concatanation of the output of multiple models.
    A feature code represents the index number of this feature in the prediction tensor.
    The output semantics of each model forms a semantic group which are listed in the dictionary semantic_groups.
    Output semantics map directly to features. Features within a group are assumed to be mutually exclusive.
    The order of the output semantics elements in each semantic group give the interpretation of the code within this group.
    '''

    def __init__(self, input_string:str, prediction:Tensor3D, semantic_groups: OrderedDict):
        # when input_string empty, length of prediciton is zero, and prediction is only Tensor2D...
        if input_string:
            assert (len(input_string) == prediction.size(2)), f"mismatch input string length {len(input_string)} and size of prediction dim zero {prediction.size(2)}"
        else:
            assert (len(input_string) == 0 and prediction.dim() == 2)
        self.input_string = input_string
        self.token_list = None
        self.prediction = prediction # character-level prediction 3D tensor
        self.semantic_groups = copy(semantic_groups)
        self.concepts = OrderedDict()
        self.scores = OrderedDict() # torch.zeros(N_group, N_token)
        self.char_level_concepts = OrderedDict()

    def decode(self, token_list):
        if token_list is None:
            self.token_list = tokenize(self.input_string)['token_list']
        else:
            self.token_list = token_list
        start_feature = 0
        for group in self.semantic_groups:
            semantics = self.semantic_groups[group]
            nf = len(semantics) # as many features as semantic output elements
            prediction_slice = self.prediction[0, start_feature:start_feature+nf, : ] # Tensor2D!
            start_feature += nf 
            self.char_level_concepts[group], self.concepts[group] , self.scores[group] = self.pred2concepts(token_list, prediction_slice, self.semantic_groups[group])

    @staticmethod
    def pred2concepts(token_list: List, prediction: Tensor3D, semantic_concepts: List) -> Tensor2D:
        '''
        Tranforms a character level multi-feature tensor into a token-level feature-code tensor.
        A feature code is the index of the feature with maximum score.
        For each token, its score is otained by averaging along the segment corresponding to its position in the text.

        Args:
            token_list: list of N Token. Each Token has start and stop attributes corresponding to its location in the text
            prediction: a 2D nf x L Tensor

        Returns:
            a 1D N Tensor with feature codes
            a 1D N Tensor with the score of each token

        '''
        L = prediction.size(1)
        N = len(token_list)
        nf= prediction.size(0)
        scores = torch.zeros(nf, N)
        for i, token in enumerate(token_list):
            for k in range(nf):
                scores[k, i] = prediction[k, token.start:token.stop].mean() # calculate score for the token by averaging the prediction over the corresponding fragment
        codes = scores.argmax(0) # the codes are the indices of features with maximum score
        token_level_scores = scores[codes.long(), range(N)] # THIS IS A BIT UNINTUITIVE: THE SCORE IS RELATIVE TO THE CLASS/CODE
        token_level_concepts = [semantic_concepts[code] for code in codes]
        char_level_concepts = [Catalogue.UNTAGGED for _ in range(L)] # initialize as untagged
        for token, concept in zip(token_list, semantic_concepts):
            char_level_concepts[token.start:token.stop] = [concept] * (token.stop - token.start)
        return char_level_concepts, token_level_concepts, token_level_scores

    def fuse_adjascent(self):
        if len(self.token_list) > 1:
            i = 0
            while i < (len(self.token_list)-1): # len(self.token_list) decreases as token are fused
                t = self.token_list[i]
                next_t = self.token_list[i+1]
                concepts = OrderedDict([(g, self.concepts[g][i]) for g in self.concepts])
                next_concepts = OrderedDict([(g, self.concepts[g][i+1]) for g in self.concepts])
                scores_spacer = OrderedDict()
                for index, key in enumerate(self.scores):
                    if next_t.start - t.stop > 0:
                        scores_spacer[key] = self.prediction[0, index, t.stop:next_t.start].mean().item()
                    else: # otherwise cannot take mean() of empty tensor
                        scores_spacer[key] = 1.0 # default such that fusion test not limited
                fuse = True
                tagged = 0
                for group in self.semantic_groups:
                    if type(concepts[group]) != type(Catalogue.UNTAGGED):
                        tagged += 1
                        fuse = fuse and (type(concepts[group]) == type(next_concepts[group])) and  (scores_spacer[group] > FUSION_THRESHOLD)
                if fuse and tagged > 0:
                    for group in self.semantic_groups:
                        self.concepts[group].pop(i+1)
                        self.scores[group][i] = (self.scores[group][i] + self.scores[group][i+1]) / 2 # oversimplification but maybe ok
                        self.char_level_concepts[group][t.stop:next_t.start] = [concepts[group]] * (next_t.start - t.stop) # does nothing if spacer empty
                    fused_text = t.text + next_t.left_spacer + next_t.text
                    fused_token = Token(fused_text, t.start, next_t.stop, len(fused_text), t.left_spacer)
                    self.token_list.pop(i+1)
                    self.token_list[i] = fused_token
                    # i stays the same since we have fused 2 tags and need to check whether to fuse further
                else:
                    i += 1 # if no fusion occured, go to next token

    def erase_with_(self, other: 'Decoder', erase_with: Tuple, target: Tuple):
        erase_with_group, erase_with_concept = erase_with
        target_group, target_concept = target
        untagged_code = Catalogue.UNTAGGED.my_index(self.semantic_groups[target_group]) # finds where the UNTAGGED feature is
        for i, (token, other_concept, my_concept) in enumerate(zip(self.token_list, other.concepts[erase_with_group], self.concepts[target_group])):
            if (type(my_concept) == type(target_concept)) and (type(other_concept) == type(erase_with_concept)): # DOES NOT WORK BUT WORKS IF type() == type() because different instances. Bad implementation ni mapper
                self.concepts[target_group][i] = Catalogue.UNTAGGED
                self.scores[target_group][i] = 0
                self.char_level_concepts[target_group][token.start:token.stop] = [Catalogue.UNTAGGED] * (token.stop - token.start)
                self.prediction[0, : , token.start:token.stop] = 0 # sets all the features to zero
                self.prediction[0, untagged_code , token.start:token.stop] = 1 # set the untagged features to 1, do we need this?

    def erase_with(self, other: 'Decoder', eraser: str, target:str) -> 'Decoder':
        cloned_self = self.clone()
        cloned_self.erase_with_(other, eraser, target)
        return cloned_self

    def cat_(self, other: 'Decoder'):
        self.semantic_groups.update(copy(other.semantic_groups))
        self.concepts.update(other.concepts)
        self.scores.update(other.scores)
        self.char_level_concepts.update(other.char_level_concepts)
        self.prediction = torch.cat([self.prediction, other.prediction.clone()], 1)

    def clone(self) -> 'Decoder':
        other = Decoder(self.input_string, self.prediction.clone(), deepcopy(self.semantic_groups))
        other.token_list = deepcopy(self.token_list)
        other.concepts = copy(self.concepts) # shallow copy
        other.scores = OrderedDict([(group, self.scores[group].clone()) for group in self.scores])
        other.char_level_concepts = OrderedDict([(group, copy(self.char_level_concepts[group])) for group in self.char_level_concepts])
        return other

class CharLevelDecoder(Decoder):

    @staticmethod
    def pred2concepts(token_list, prediction, semantic_concepts):
        '''
        Here we take first the argmax at character level to obtain the classification of each character. 
        '''
        N = len(token_list)
        nf = prediction.size(0)
        char_level_codes = prediction.argmax(0)
        char_level_concepts = [semantic_concepts[code] for code in char_level_codes]
        scores = torch.zeros(nf, N)
        for i, token in enumerate(token_list):
            for k in range(nf):
                scores[k, i] = prediction[k, token.start:token.stop].mean() # calculate score for the token by averaging the prediction over the corresponding fragment
        token_level_codes = scores.argmax(0) # the codes are the indices of features with maximum score
        token_level_scores = scores[token_level_codes.long(), range(N)] # THIS IS A BIT UNINTUITIVE: THE SCORE IS RELATIVE TO THE CLASS/CODE
        token_level_concepts = [semantic_concepts[code] for code in token_level_codes]
        return char_level_concepts, token_level_concepts, token_level_scores

