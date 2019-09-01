# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import os
import torch
import argparse
import numpy as np
from random import randrange
from torch.utils.data import DataLoader
from torch import nn
from .dataset import Data4th
from .trainer import predict_fn, collate_fn
from ..predict.decode import Decoder
from ..common.progress import progress
from ..common.importexport import load_model
from ..common.utils import timer
from ..common.options import Options
from .. import config

DEFAULT_THRESHOLD = config.default_threshold

class Accuracy(object):

    def __init__(self, model, minibatches, nf_output, tokenize=False):
        self.model = model
        self.minibatches = minibatches
        self.N = len(self.minibatches) * self.minibatches.batch_size
        self.nf = nf_output
        self.tokenize  = tokenize
        # self.target_concepts = []
        # if torch.cuda.is_available(): # or torch.cuda.is_available() ?
        #     self.cuda_on = True
        # if self.tokenize:
        #     for i, m in enumerate(self.minibatches):
        #         progress(i, self.minibatches.minibatch_number, "tokenizing minibatch {}".format(i))
        #         m_output = m.output
        #         if self.cuda_on:
        #             m_output = m_output.cuda()
        #         m.add_token_lists()
        #         d = Decoded(m.text, m_output, self.model.output_semantics)
        #         d.decode_with_token(m.tokenized)
        #         self.target_concepts.append(d.concepts)
        # epsilon = 1e-12
        # thresh = [concept.threshold for concept in self.model.output_semantics]
        # self.thresholds = torch.Tensor(thresh).resize_(1, self.nf , 1 )
        # if self.cuda_on:
        #     self.thresholds = self.thresholds.cuda()
        #     self.target_concepts = [b.cuda() for b in self.target_concepts]

    @timer
    def run(self, predict_fn):
        loss_avg = 0
        p_sum = torch.zeros(self.nf)
        tp_sum = torch.zeros(self.nf)
        fp_sum = torch.zeros(self.nf)
        if torch.cuda.is_available():
            p_sum = p_sum.cuda()
            tp_sum = tp_sum.cuda()
            fp_sum = fp_sum.cuda()
        for i, m in enumerate(self.minibatches):
            progress(i, len(self.minibatches), "\tevaluating model                              ")
            x, y, y_hat, loss = predict_fn(self.model, m, eval=True)
            # if self.tokenize:
            #     prediction_decoded = Decoded(m.text, prediction, self.model.output_semantics)
            #     prediction_decoded.decode_with_token(m.tokenized)
            #     p, tp, fp = self.tpfp(prediction_decoded.concepts, self.target_concepts[i])
            # else:
            loss_avg += loss.cpu().data
            p, tp, fp = self.tpfp(y_hat, y)
            p_sum += p
            tp_sum += tp
            fp_sum += fp
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / p_sum
        f1 = 2 * recall * precision / (recall + precision)
        loss_avg = loss_avg / self.N
        return precision.cpu(), recall.cpu(), f1.cpu(), loss

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

    def __init__(self, model_basename, testset_basenames):#, tokenize):
        self.model_name = model_basename
        self.model = load_model(model_basename)
        self.output_semantics = self.model.output_semantics
        self.opt = self.model.opt
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "GPUs available.")
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
            self.model.output_semantics = self.output_semantics

        # self.tokenize = tokenize
        self.opt.data_path_list = [os.path.join(config.data4th_dir, f) for f in testset_basenames] # it has to be a list (to allow joint training on multiple datasets)
        testset = Data4th(self.opt, 'test')
        testset = DataLoader(testset, batch_size=self.opt.minibatch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, drop_last=True, timeout=60)
        benchmark = Accuracy(self.model, testset, self.opt.nf_output)#, tokenize=self.tokenize)
        self.precision, self.recall, self.f1, self.loss = benchmark.run(predict_fn)

    def display(self):
        print("\n\n\033[31;1m========================================================\033[0m")
        print(f"\n\033[31;1m Data: {' + '.join(self.opt.data_path_list)}\033[0m")
        print(f"\n\033[31;1m Model: {self.model_name}\033[0m")
        print("\n Global stats: \033[1m\n")
        print(f"\t\033[32;1mprecision\033[0m = {self.precision.mean()}")
        print(f"\t\033[33;1mrecall\033[0m = {self.recall.mean():.2f}")
        print(f"\t\033[36;1mf1\033[0m = {self.f1.mean():.2f}")

        for i, feature in enumerate(self.model.output_semantics):
            print(f"\n Feature: '\033[1m{feature}\033[0m'\n")
            print(f"\t\033[32;1mprecision\033[0m = {self.precision[i]:.2f}")
            print(f"\t\033[33;1mrecall\033[0m = {self.recall[i]:.2f}")
            print(f"\t\033[36;1mf1\033[0m = {self.f1[i]:.2f}")

# class ScanThreshold():

#     def __init__(self, model_basename, dataset_basename, tokenize):
#         self.model = load_model(model_basename)
#         self.opt = self.model.opt
#         print("opt from model", "; ".join(["{}={}".format(k, str(self.opt[k])) for k in self.opt]))
#         self.tokenize = tokenize
#         loader = Loader(self.opt) # validation_fraction == 0 ==> dataset['single'] is loaded
#         validation = loader.prepare_datasets(os.path.join(config.data4th_dir, dataset_basename, 'valid'))
#         self.minibatches = Minibatches(validation, int(self.opt['minibatch_size']))


#     def run(self, n=11):
#         old_thresholds = [concept.threshold for concept in self.model.output_semantics]
#         for threshold in np.linspace(0, 1, n):
#             for i, _ in enumerate(self.model.output_semantics):
#                 self.model.output_semantics[i].threshold = threshold
#             evaluator = Accuracy(self.model, self.minibatches, tokenize=self.tokenize) # evaluator needs to be redone because thresholds need to be reassigned
#             precision, recall, f1 = evaluator.run()
#             self.show(precision, recall, f1)
#         for i, _ in enumerate(self.model.output_semantics):
#                 self.model.output_semantics[i].threshold = old_thresholds[i]

    # def show(self, precision, recall, f1):
    #     precision = "; ".join(["{:.3f}".format(p.item()) for p in precision])
    #     recall = "; ".join(["{:.3f}".format(p.item()) for p in recall])
    #     f1 = "; ".join(["{:.3f}".format(p.item()) for p in f1])
    #     print("\t".join(["{:.3f}".format(c.threshold) for c in self.model.output_semantics]), precision, recall, f1)

def main():
    parser = config.create_argument_parser_with_defaults(description='Accuracy evaluation.')
    parser.add_argument('filenames', help='Basename(s) of the dataset(s) to import (testset)')
    parser.add_argument('model', help='Basename of the model to benchmark.')
    # parser.add_argument('-T' , '--no_token', action='store_true', help='Flag to disable tokenization.')
    # parser.add_argument('-S' , '--scan', action='store_true', help='Flag to switch to threshold scaning mode.')

    arguments = parser.parse_args()
    basenames = [f.strip() for f in arguments.filename.split(',')]
    model_basename = arguments.model
    # scan_threshold = arguments.scan
    # tokenize = not arguments.no_token
    print(f"model: {model_basename}, testset: {'-'.join(basenames)}")
    # if scan_threshold:
    #     s = ScanThreshold(model_basename, basename)#, tokenize=tokenize)
    #     s.run()
    # else:
    b = Benchmark(model_basename, basenames)#, tokenize=tokenize)
    b.display()

if __name__ == '__main__':
    main()
