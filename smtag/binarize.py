import torch
import re
from smtag.utils import xml_escape


class Binarized:
    def __init__(self, text_examples, prediction, output_semantics): # will need a concept map to set feature-specific thresholds Object Prediction with inputstring, input concept output concept and output tensor
        self.text_examples = text_examples
        self.prediction = prediction
        self.output_semantics = output_semantics
        self.N = prediction.size(0)
        self.nf = prediction.size(1)
        self.L = prediction.size(2)
        dim = (self.N, self.nf, self.L)
        self.binarized_pred = torch.zeros(dim)
        self.start = torch.zeros(dim)
        self.stop = torch.zeros(dim)
        self.marks = torch.zeros(dim)
        self.score = torch.zeros(dim)
        self.tokenized = []

    def binarize_with_token(self, tokenized_examples):
        self.tokenized = tokenized_examples
        for i in range(self.N):
            token = tokenized_examples[i]
            for t in token:
                start = t.start
                stop = t.stop
                token_length = stop - start
                for k in range(self.nf):
                    avg_score = float(self.prediction[i, k, start:stop].sum()) / token_length
                    #feature_name = attrmap[k][1]
                    #local threshold = CONFIG.THRESHOLDS[feature_name].word_score
                    threshold = 0.5 # OVERSIMPLIFICATION; NEEDS TO BE ADJUSTABLE AND FEATURE-specific
                    if avg_score >= threshold: 
                        self.start[i, k, start] = 1
                        self.stop[i, k, stop-1] = 1 # should stop mark be stop-1 to be the last character and not the next? otherwise need to test if stop < length: 
                        self.score[i, k, start] = round(100*avg_score)
                        self.marks[i, k, start:stop].fill_(1)
    
    def fuse_adjascent(self, regex=" "):

        test = re.compile(regex)
        for i in range(self.N):
            input_string = self.text_examples[i]
            #if self.tokenized: 
            #    pos_iter = PositionIter(token=self.tokenized[i], mode='stop')
            #else:
            #    pos_iter = PositionIter(input_string)
            #for pos, _ in pos_iter:
            for t in self.tokenized[i]:
                pos = t.stop
                #s = "this is the black cat"
                #                 ||||| |||
                #     0123456789012345678901
                #                 ^    |       s[start:stop] => s[12:17] == 'black'
                #                       ^  |   s[start:stop] => s[18:21] == 'cat'
                # pos = 17: stop[, ,17] > 0.99 and start[,,18] > 0.99 [17]==" "
                for k in range(self.nf):
                    if self.binarized_pred.stop[i, k, pos] > 0.99 and self.binarized_pred.start[1, i, pos+1] > 0.99 and re.match(test, input_string[pos]): 
                        self.binarized_pred.stop[i, k, pos] = 0 # remove the stop boundary of first term
                        self.binarized_pred.start[i, k, pos+1] = 0 # remove the start boundary of next term
                        self.binarized_pred.marks[i, k, pos] = 1 # fill the gap by marking the space as part of the tagged term
                        self.binarized_pred.score[i, k, pos+1] = 0 #ideally the average of the score of the fused words but would ned to find start of upstream word 
                        #cannot fuse token because other predictions may not fuse? how likely is this to happen?; makes it obligatory to use char-by-char serializer; will be slower

