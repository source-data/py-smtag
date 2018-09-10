# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import torch
from .minibatches import Minibatches
from .loader import Loader
from ..predict.binarize import Binarized
from ..common.progress import progress
from ..common.config import DEFAULT_THRESHOLD
from ..common.importexport import load_model

class Accuracy(object):
 

    def __init__(self, model, minibatches, tokenize=True):
        self.model = model # is this just a reference? if yes, no need to pass model every time
        self.minibatches = minibatches
        self.tokenize  = tokenize
        if self.tokenize:
            for i, m in enumerate(self.minibatches):
                progress(i, self.minibatches.minibatch_number, "tokenizing minibatch {}".format(i))
                m.add_token_lists()
        if torch.cuda.device_count() > 0: # or torch.cuda.is_available() ?
            self.cuda_on = True
            print("{} GPUs available!".format(torch.cuda.device_count()))
        else:
            self.cuda_on = False

    def run(self):

        nf = self.minibatches.nf_output
        epsilon = 1e-12
        p_sum = torch.Tensor(nf).fill_(epsilon)
        tp_sum = torch.Tensor(nf).fill_(epsilon)
        fp_sum = torch.Tensor(nf).fill_(epsilon)

        if self.cuda_on:
            p_sum = p_sum.cuda()
            tp_sum = tp_sum.cuda()
            fp_sum = fp_sum.cuda()

        output_semantics = self.model.output_semantics

        for m in self.minibatches:
            self.model.eval()
            prediction = self.model(m.input)
            if self.tokenize: 
                bin_pred = Binarized(m.text, prediction, output_semantics)
                bin_pred.binarize_with_token(m.tokenized)
                bin_target = Binarized(m.text, m.output, output_semantics)
                bin_target.binarize_with_token(m.tokenized)
                p, tp, fp = self.tpfp(bin_pred.start, bin_target.start)
            else:
                p, tp, fp = self.tpfp(prediction, m.output, DEFAULT_THRESHOLD)

            p_sum += p
            tp_sum += tp
            fp_sum += fp

        self.model.train()
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / p_sum
        f1 = 2 * recall * precision / (recall + precision)

        return precision, recall, f1

    @staticmethod
    def tpfp(prediction, target, threshold=0.99):
        """
        Computes accuracy at the level of individual characters.

        Args:
            threshold (0..1): to call a character as tagged ('hit')
            prediction (4D tensor): the prediction for which the accuracy has to be computed
            target (4D tensor): the target to which the prediction has to be compared to 

        Returns:
            Three 1D tensors with the number per feature of positives, true positives and false positives, respectively
        """
        # https://en.wikipedia.org/wiki/Precision_and_recall
        nf = target.size(1)
        p = target.sum(2).sum(0) # sum over the characters on a line and sum of these over all examples of the batch
        predx = (prediction - threshold).clamp(0).ceil() # threshold prediction to binarize in 0 (no hit) or 1 (hit)
        fp = (predx - target).clamp(0).sum(2).sum(0) # false positives when prediction is 1 but target is 0
        tp = predx.sum(2).sum(0) - fp 
        return p.view(nf), tp.view(nf), fp.view(nf) # output as 1 dim tensor with one metric per output feature

class Benchmark():

    def __init__(self, model_basename, testset_basename, tokenize):
        self.model = load_model(model_basename)
        self.opt = self.model.opt
        self.opt['validation_fraction'] = 0 # this will set the dataset mode into testset mode
        self.tokenize = tokenize
        loader = Loader(self.opt)
        testset = loader.prepare_datasets(testset_basename)
        testset = Minibatches(testset['test'], self.opt['minibatch_size'])
        benchmark = Accuracy(self.model, testset, tokenize=self.tokenize)
        self.precision, self.recall, self.f1 = benchmark.run()

    def display(self):
        print("\n Global stats: \27[1m")
        print("\t\27[32;1mprecision\27[0m = {}.2f".format(self.precision.mean()))
        print("\t\27[33;1mrecall\27[0m = {}.2f".format(self.recall.mean()))
        print("\t\27[36;1mf1\27[0m = {}.2f".format(self.f1.mean()))

        for i, feature in enumerate(self.model.output_semantics):
                print("\n Feature: '\27[1m{}\27[0m'\n".format(feature))
                print("\t\27[32;1mprecision\27[0m = {}.2f".format(self.precision[i]))
                print("\t\27[33;1mrecall\27[0m = {}.2f".format(self.recall[i]))
                print("\t\27[36;1mf1\27[0m = {}.2f".format(self.f1[i]))