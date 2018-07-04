# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
from math import ceil
from collections import namedtuple
from smtag.converter import TString
from smtag.binarize import Binarized
from smtag.serializer import Serializer
from smtag.utils import tokenize, timer
from smtag.operations import t_replace
from smtag.config import MIN_PADDING,  MIN_SIZE, MARKING_CHAR


MARKING_ENCODED = TString(MARKING_CHAR)
SPACE_ENCODED = TString(" ")

class Predictor: #(nn.Module?)

    def __init__(self, model, tag='sd-tag', format='xml'):
        self.model = model
        self.tag = tag
        self.format = format

    def padding(self, input): 
        padding_length = ceil(max(MIN_SIZE - len(input), MIN_PADDING)/2)
        pad = SPACE_ENCODED.repeat(padding_length)
        return pad + input + pad

    def combine_with_input_features(self, input, additional_input_features=None):
        #CAUTION: should additional_input_features be cloned before modifying it?
        if additional_input_features is not None:
            nf2 = additional_input_features.size(1)
            padding_length = (len(input) - additional_input_features.size(2)) / 2
            pad = torch.zeros(1, nf2, padding_length)
            padded_additional_input_features = torch.cat([pad, additional_input_features, pad], 2) # flanking the encoded string with padding tensor left and right along third dimension
            combined_input = torch.cat([input, padded_additional_input_features], 1) # adding additional inputs under the encoded string along second dimension
        else:
            combined_input = input
        return combined_input

    def forward(self, input, additional_input_features = None): 
        padded = self.padding(input)
        L = len(padded)
        padding_length = int((L - len(input)) / 2)
        input = self.combine_with_input_features(padded, additional_input_features)
        
        #PREDICTION
        self.model.eval()
        prediction = self.model(input.toTensor()) #.float()
        self.model.train()

        #remove safety padding
        prediction = prediction[ : , : , padding_length : L-padding_length]
        return prediction

    def pred_bin(self, input_string, prediction, output_semantics):
        bin_pred = Binarized([str(input_string)], prediction, output_semantics) # this is where the transfer of the output semantics from the model to the binarized prediction happen; will be used for serializing
        token_list = tokenize(str(input_string)) # needs to be string, change TString to keep string?
        bin_pred.binarize_with_token([token_list])
        bin_pred.fuse_adjascent(regex=" ")
        return bin_pred


class SimplePredictor(Predictor):

    def __init__(self, model, tag='sd-tag', format='xml'):
        super(SimplePredictor, self).__init__(model) # the model should carry the semantics with him, transmit it to pred and be used by Tagger.element()

    def pred_binarized(self, input_t_string, output_semantics):
        prediction = self.forward(input_t_string)
        return super(SimplePredictor, self).pred_bin(input_t_string, prediction, output_semantics)


class ContextualPredictor(Predictor):

    def __init__(self, model, tag='sd-tag', format='xml'):
        super(ContextualPredictor, self).__init__(model)
        self.tag = tag
        self.format = format

    def anonymize(self, input, marks, replacement = MARKING_CHAR):
        i = 0
        res = ''
        for c in str(input):
            if marks[0, i] > 0.99:
                res += replacement
            else:
                res += c
            i += 1
        return TString(res)
        # return TString(t_replace(input.toTensor().clone(), marks, replacement.toTensor())) # cute but surprisingly slow!

    def forward(self, input, marks): 
        predictions = []
        # each anonymization needs to be done separately as it anonymize all the input features
        for i in range(marks.size(1)):
            anonymized_t = self.anonymize(input, marks[ : , i, : ]) # marks[ : , i, : ] is then 2D N x L
            predictions.append(super(ContextualPredictor, self).forward(anonymized_t))
        prediction = torch.cat(predictions, 1)
        return prediction

    def pred_binarized(self, input_t_string, marks, output_semantics):
        prediction = self.forward(input_t_string, marks)
        return super(ContextualPredictor, self).pred_bin(input_t_string, prediction, output_semantics)