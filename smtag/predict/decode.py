# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
from typing import List, Tuple
from collections import OrderedDict
from copy import copy, deepcopy
from ..common.converter import StringList
from ..common.utils import tokenize, Token
from ..common.mapper import Catalogue, Concept
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

    def __init__(self, input_strings: StringList, prediction:Tensor3D, semantic_groups: OrderedDict):
        # when input_string empty, length of prediciton is zero, and prediction is only Tensor2D...
        self.L = len(input_strings)
        self.N = input_strings.depth
        if input_strings:
            try:
                assert (self.N == prediction.size(0)) and (self.L == prediction.size(2)), f"Size mismatch: input string has {self.N} examples with length {self.L}, prediction tensor is {prediction.size(0)} x {prediction.size(1)} x {prediction.size(2)}"
            except AssertionError as e:
                print(e)
                print(input_strings.words)
                import pdb; pdb.set_trace()
        else:
            assert (len(input_strings) == 0 and prediction.dim() == 2)
        self.input_strings = input_strings
        self.token_lists = []
        self.prediction = prediction # character-level prediction 3D tensor
        self.semantic_groups = copy(semantic_groups)
        self.concepts = []
        self.scores = []
        self.char_level_concepts = []


    def decode(self, token_lists: List[List[Token]]=None):
        if token_lists is None:
            self.token_lists = [tokenize(s)['token_list'] for s in self.input_strings]
            assert len(self.token_lists) ==  self.N, f"number of examples in token list different from examples in tensor and input_strings: {self.token_lists} != {self.N}"
        else:
            self.token_lists = token_lists
        for n, token_list in enumerate(self.token_lists):
            start_feature = 0
            char_level_concepts = OrderedDict()
            concepts = OrderedDict()
            scores = OrderedDict()
            for group in self.semantic_groups:
                semantics = self.semantic_groups[group]
                nf = len(semantics) # as many features as semantic output elements
                # prediction_slice = self.prediction[n, start_feature:start_feature+nf, : ] # Tensor2D! 
                char_level_concepts[group], concepts[group] , scores[group] = self.pred2concepts(n, start_feature, nf, token_list, self.semantic_groups[group])
                start_feature += nf
            self.char_level_concepts.append(char_level_concepts)
            self.concepts.append(concepts)
            self.scores.append(scores)

    @staticmethod
    def compute_score(p):
        l = p.tolist()
        score = sum(l)/len(l) # faster than tensor.mean()
        # score = p.mean() # slow!
        return score

    def get_max_scores(self, token, example_index, starting_feature, nf):
        max_score_value = 0
        max_score_index = 0
        for k in range(nf):
            score = self.compute_score(self.prediction[example_index, starting_feature+k, token.start:token.stop])
            if score > max_score_value:
                max_score_value = score
                max_score_index = k
        return max_score_index, max_score_value

    def scan_token_list(self, N, token_list, example_index, starting_feature, nf):
        index_of_max_score = [0] * N
        max_scores = [0] * N
        for i, token in enumerate(token_list):
            index_of_max_score[i],  max_scores[i] = self.get_max_scores(token, example_index, starting_feature, nf)
        return index_of_max_score, max_scores
    
    def pred2concepts(self, example_index, starting_feature, nf, token_list: List[Token], semantic_concepts: List[Concept]):
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

        L = self.prediction.size(2)
        N = len(token_list)

        codes, token_level_scores = self.scan_token_list(N, token_list, example_index, starting_feature, nf)
        token_level_concepts = [semantic_concepts[code] for code in codes]
        char_level_concepts = [Catalogue.UNTAGGED for _ in range(L)] # initialize as untagged
        for token, concept in zip(token_list, semantic_concepts):
            char_level_concepts[token.start:token.stop] = [concept] * (token.stop - token.start)
        return char_level_concepts, token_level_concepts, token_level_scores

    def fuse_adjacent(self):
        for n in range(self.N):
            if len(self.token_lists[n]) > 1:
                i = 0
                while i+1 < (len(self.token_lists[n])): # note: len(self.token_list) decreases as token are fused
                    t = self.token_lists[n][i]
                    try:
                        next_t = self.token_lists[n][i+1]
                    except IndexError:
                        import pdb; pdb.set_trace()
                    concepts = OrderedDict([(g, self.concepts[n][g][i]) for g in self.concepts[n]])
                    next_concepts = OrderedDict([(g, self.concepts[n][g][i+1]) for g in self.concepts[n]])
                    scores_spacer = OrderedDict()
                    for index, key in enumerate(self.scores[n]):
                        if next_t.start - t.stop > 0:
                            l = self.prediction[n, index, t.stop:next_t.start].tolist()
                            scores_spacer[key] = sum(l) / len(l)
                        else: # otherwise cannot take mean() of empty tensor
                            scores_spacer[key] = 1.0 # default such that fusion test not limited
                    fuse = True
                    tagged = 0
                    for group in self.semantic_groups:
                        if concepts[group] != Catalogue.UNTAGGED:
                            tagged += 1
                            fuse = fuse and (concepts[group] == next_concepts[group]) and  (scores_spacer[group] > FUSION_THRESHOLD)
                    if fuse and tagged > 0:
                        for group in self.semantic_groups:
                            self.concepts[n][group].pop(i+1)
                            self.scores[n][group][i] = (self.scores[n][group][i] + self.scores[n][group][i+1]) / 2 # oversimplification but maybe ok
                            self.char_level_concepts[n][group][t.stop:next_t.start] = [concepts[group]] * (next_t.start - t.stop) # does nothing if spacer empty
                        fused_text = t.text + next_t.left_spacer + next_t.text
                        fused_token = Token(fused_text, t.start, next_t.stop, len(fused_text), t.left_spacer)
                        self.token_lists[n].pop(i+1)
                        self.token_lists[n][i] = fused_token
                        # i stays the same since we have fused 2 tags and need to check whether to fuse further
                    else:
                        i += 1 # if no fusion occured, go to next token

    @staticmethod
    def check_compatible_depth(a, b):
        assert a.N == b.N, f"depth mismatch: {a.N} != {b.N}"

    def erase_with_(self, other: 'Decoder', erase_with: Tuple, target: Tuple): # in place
        self.check_compatible_depth(self, other)
        erase_with_group, erase_with_concept = erase_with
        target_group, target_concept = target
        untagged_code = self.semantic_groups[target_group].index(Catalogue.UNTAGGED) # finds where the UNTAGGED feature is
        for n in range(self.N):
            for i, (token, other_concept, my_concept) in enumerate(zip(self.token_lists[n], other.concepts[n][erase_with_group], self.concepts[n][target_group])):
                if (my_concept == target_concept) and (other_concept == erase_with_concept):
                    self.concepts[n][target_group][i] = Catalogue.UNTAGGED
                    self.scores[n][target_group][i] = 0
                    self.char_level_concepts[n][target_group][token.start:token.stop] = [Catalogue.UNTAGGED] * (token.stop - token.start)
                    self.prediction[n, : , token.start:token.stop] = 0 # sets all the features to zero
                    self.prediction[n, untagged_code , token.start:token.stop] = 1 # set the untagged features to 1, do we need this?

    def erase_with(self, other: 'Decoder', eraser: str, target:str) -> 'Decoder': # after cloning
        self.check_compatible_depth(self, other)
        cloned_self = self.clone()
        cloned_self.erase_with_(other, eraser, target)
        return cloned_self

    def cat_(self, other: 'Decoder'):
        self.check_compatible_depth(self, other)
        self.semantic_groups.update(copy(other.semantic_groups))
        for n in range(self.N):
            self.concepts[n].update(other.concepts[n])
            self.scores[n].update(other.scores[n])
            self.char_level_concepts[n].update(other.char_level_concepts[n])
        self.prediction = torch.cat([self.prediction, other.prediction], 1)

    def clone(self) -> 'Decoder':
        other = Decoder(self.input_strings.clone(), self.prediction.clone(), deepcopy(self.semantic_groups))
        other.token_lists = deepcopy(self.token_lists)
        other.concepts = deepcopy(self.concepts) # shallow copy
        other.scores = deepcopy(self.scores) # why was this not used in previous version?
        other.char_level_concepts = deepcopy(self.char_level_concepts)
        # for n in range(self.N):
        #     other.scores[n] = OrderedDict([(group, self.scores[n][group].clone()) for group in self.scores[n]])
        #     other.char_level_concepts[n] = OrderedDict([(group, deepcopy(self.char_level_concepts[n][group])) for group in self.char_level_concepts[n]])
        return other

class CharLevelDecoder(Decoder):

    def pred2concepts(self, example_index, starting_feature, nf, token_list: List[Token], semantic_concepts: List[Concept]):
        '''
        Here we take first the argmax at character level to obtain the classification of each character. 
        '''
        N = len(token_list)

        char_level_codes = self.prediction[example_index, starting_feature:starting_feature+nf, : ] # 2D slice!
        char_level_codes = char_level_codes.argmax(0)
        char_level_concepts = [semantic_concepts[code] for code in char_level_codes]
        codes, token_level_scores = self.scan_token_list(N, token_list, example_index, starting_feature, nf)
        token_level_concepts = [semantic_concepts[code] for code in codes]
        return char_level_concepts, token_level_concepts, token_level_scores

