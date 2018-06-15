# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
from math import ceil
from collections import namedtuple
from smtag.converter import TString
from smtag.binarize import Binarized
from smtag.serializer import Serializer
from smtag.utils import tokenize
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
        prediction = self.model(input.toTensor())
        self.model.train()
    
        #remove safety padding
        prediction = prediction[ : , : , padding_length : L-padding_length]
        return prediction

    def serialize(self, input_string, prediction):
        bin_pred = Binarized([input_string], prediction, self.model.output_semantics) # this is where the transfer of the output semantics from the model to the binarized prediction happen; will be used for serializing
        token_list = tokenize(input_string)
        bin_pred.binarize_with_token([token_list])
        bin_pred.fuse_adjascent(regex=" ")
        tagger = Serializer(self.tag, self.format)
        tagged_ml_string = tagger.serialize(bin_pred)
        return tagged_ml_string


class EntityPredictor(Predictor):
    
    def __init__(self, model, tag='sd-tag', format='xml'):
        super(EntityPredictor, self).__init__(model) # the model should carry the semantics with him, transmit it to pred and be used by Tagger.element()


    def markup(self, input_string):
        input = TString(input_string)
        prediction= self.forward(input)
        return super(EntityPredictor, self).serialize(input_string, prediction)

class SemanticsFromContextPredictor(Predictor):
    
    def __init__(self, model, tag='sd-tag', format='xml'):
        super(SemanticsFromContextPredictor, self).__init__(model)
        self.tag = tag
        self.format = format

    def anonymize(self, input, marks, replacement = MARKING_ENCODED):
        
        return TString(t_replace(input.toTensor().clone(), marks, replacement.toTensor()))

    def forward(self, input, marks): 
        predictions = []
        for i in range(marks.size(1)):
            anonymized_t = self.anonymize(input, marks[ : , i, : ]) # marks is then 2D N x L
            predictions.append(super(SemanticsFromContextPredictor, self).forward(anonymized_t))
        prediction = torch.cat(predictions, 1)
        return prediction
    
    def markup(self, input_string, binarized_entities):
        input = TString(input_string)
        prediction = self.forward(input, binarized_entities)
        return super(SemanticsFromContextPredictor, self).serialize(input_string, prediction)