# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import os
import torch
import argparse
import numpy as np
from random import randrange
from .minibatches import Minibatches
from .loader import Loader
from ..predict.decode import Decoder
from ..common.progress import progress
from ..common.importexport import load_model
from .. import config

DEFAULT_THRESHOLD = config.default_threshold

class Accuracy(object):

    def __init__(self, model, minibatches, tokenize=False):
        self.model = model # is this just a reference? if yes, no need to pass model every time
        self.minibatches = minibatches
        self.nf = self.minibatches.nf_output
        self.tokenize  = tokenize
        self.bin_target_start = []
        if torch.cuda.device_count() > 0: # or torch.cuda.is_available() ?
            self.cuda_on = True
        else:
            self.cuda_on = False
        # if self.tokenize:
        #     for i, m in enumerate(self.minibatches):
        #         progress(i, self.minibatches.minibatch_number, "tokenizing minibatch {}".format(i))
        #         m_output = m.output
        #         if self.cuda_on:
        #             m_output = m_output.cuda()
        #         m.add_token_lists()
        #         b = Binarized(m.text, m_output, self.model.output_semantics)
        #         b.binarize_with_token(m.tokenized)
        #         self.bin_target_start.append(b.start)
        # epsilon = 1e-12
        # thresh = [concept.threshold for concept in self.model.output_semantics]
        # self.thresholds = torch.Tensor(thresh).resize_(1, self.nf , 1 )
        # if self.cuda_on:
        #     self.thresholds = self.thresholds.cuda()
        #     self.bin_target_start = [b.cuda() for b in self.bin_target_start]

    def run(self):
        p_sum = torch.zeros(self.nf)
        tp_sum = torch.zeros(self.nf)
        fp_sum = torch.zeros(self.nf)
        if torch.cuda.is_available():
            p_sum = p_sum.cuda()
            tp_sum = tp_sum.cuda()
            fp_sum = fp_sum.cuda()
        for m in self.minibatches:
            m_input = m.input
            m_output = m.output
            if torch.cuda.is_available():
                m_input = m_input.cuda()
                m_output = m_output.cuda()
            with torch.no_grad():
                self.model.eval()
                prediction = self.model(m_input)
                self.model.train()
            # if self.tokenize: 
            #     bin_pred = Binarized(m.text, prediction, self.model.output_semantics)
            #     bin_pred.binarize_with_token(m.tokenized)
            #     p, tp, fp = self.tpfp(bin_pred.start, self.bin_target_start[i])
            # else:
            p, tp, fp = self.tpfp(prediction, m_output)
            p_sum += p
            tp_sum += tp
            fp_sum += fp
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / p_sum
        f1 = 2 * recall * precision / (recall + precision)
        return precision, recall, f1

    @staticmethod
    def tpfp(prediction, target):
        """
        Args:
            prediction (3D Tensor): predicted class features
            target (3D Tensor): target classes
        """

        nf = prediction.size(1)
        cond_p = torch.zeros(nf).to(torch.float)
        pred_p = torch.zeros(nf).to(torch.float)
        tp = torch.zeros(nf).to(torch.float)
        fp = torch.zeros(nf).to(torch.float)
        if torch.cuda.is_available():
            cond_p = cond_p.cuda()
            pred_p = pred_p.cuda()
            tp = tp.cuda()
            fp = fp.cuda()
        predicted_classes = prediction.argmax(1)
        target_classes = target.argmax(1)
        for f in range(nf):
            cond_pos = (target_classes == f)
            pred_pos = (predicted_classes == f)
            true_pos = cond_pos * pred_pos # element-wise multiply ByteTensors to find overlap
            cond_p[f] = cond_pos.sum()
            pred_p[f] = pred_pos.sum()
            tp[f] = true_pos.sum()
            fp[f] =  pred_p[f] - tp[f]
        return cond_p, tp, fp


class Benchmark():

    def __init__(self, model_basename, dataset_basename, tokenize):
        self.model_name = model_basename
        self.model = load_model(model_basename)
        self.opt = self.model.opt
        self.tokenize = tokenize
        loader = Loader(self.opt)
        self.testset_name = os.path.join(config.data4th_dir, dataset_basename,'test')
        testset = loader.prepare_datasets(self.testset_name)
        minibatches = Minibatches(testset, self.opt['minibatch_size'])
        benchmark = Accuracy(self.model, minibatches, tokenize=self.tokenize)
        self.precision, self.recall, self.f1 = benchmark.run()

    def display(self):
        print("\n\n\033[31;1m========================================================\033[0m")
        print("\n\033[31;1m Data: {}\033[0m".format(self.testset_name))
        print("\n\033[31;1m Model: {}\033[0m".format(self.model_name))
        print("\n Global stats: \033[1m\n")
        print("\t\033[32;1mprecision\033[0m = {}.2f".format(self.precision.mean()))
        print("\t\033[33;1mrecall\033[0m = {}.2f".format(self.recall.mean()))
        print("\t\033[36;1mf1\033[0m = {}.2f".format(self.f1.mean()))

        for i, feature in enumerate(self.model.output_semantics):
            print("\n Feature: '\033[1m{}\033[0m'\n".format(feature))
            print("\t\033[32;1mprecision\033[0m = {}.2f".format(self.precision[i]))
            print("\t\033[33;1mrecall\033[0m = {}.2f".format(self.recall[i]))
            print("\t\033[36;1mf1\033[0m = {}.2f".format(self.f1[i]))

class ScanThreshold():

    def __init__(self, model_basename, dataset_basename, tokenize):
        self.model = load_model(model_basename)
        self.opt = self.model.opt
        print("opt from model", "; ".join(["{}={}".format(k, str(self.opt[k])) for k in self.opt]))
        self.tokenize = tokenize
        loader = Loader(self.opt) # validation_fraction == 0 ==> dataset['single'] is loaded
        validation = loader.prepare_datasets(os.path.join(config.data4th_dir, dataset_basename, 'valid'))
        self.minibatches = Minibatches(validation, int(self.opt['minibatch_size']))


    def run(self, n=11):
        old_thresholds = [concept.threshold for concept in self.model.output_semantics]
        for threshold in np.linspace(0, 1, n):
            for i, _ in enumerate(self.model.output_semantics):
                self.model.output_semantics[i].threshold = threshold
            evaluator = Accuracy(self.model, self.minibatches, tokenize=self.tokenize) # evaluator needs to be redone because thresholds need to be reassigned
            precision, recall, f1 = evaluator.run()
            self.show(precision, recall, f1)
        for i, _ in enumerate(self.model.output_semantics):
                self.model.output_semantics[i].threshold = old_thresholds[i]

    def show(self, precision, recall, f1):
        precision = "; ".join(["{:.3f}".format(p.item()) for p in precision])
        recall = "; ".join(["{:.3f}".format(p.item()) for p in recall])
        f1 = "; ".join(["{:.3f}".format(p.item()) for p in f1])
        print("\t".join(["{:.3f}".format(c.threshold) for c in self.model.output_semantics]), precision, recall, f1)

def main():
    parser = argparse.ArgumentParser(description='Accuracy evaluation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', default='test_entities_test', help='Basename of the dataset to import (testset)')
    parser.add_argument('-m' , '--model',  default='entities.sddl', help='Basename of the model to benchmark.')
    parser.add_argument('-T' , '--no_token', action='store_true', help='Flag to disable tokenization.')
    parser.add_argument('-S' , '--scan', action='store_true', help='Flag to switch to threshold scaning mode.')

    arguments = parser.parse_args()
    basename = arguments.file
    model_basename = arguments.model
    scan_threshold = arguments.scan
    tokenize = not arguments.no_token
    print("model: {}, testset: {}, tokenization: {}".format(model_basename, basename, tokenize))
    if scan_threshold:
        s = ScanThreshold(model_basename, basename, tokenize=tokenize)
        s.run()
    else:
        b = Benchmark(model_basename, basename, tokenize=tokenize)
        b.display()

if __name__ == '__main__':
    main()