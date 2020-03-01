# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import math
import torch
from torch.nn import functional as F
from random import random
from .converter import TString
# from ..train.evaluator import Accuracy
from ..train.builder import SmtagModel
from ..train.dataset import Data4th, BxL, BxCxL
from ..predict.predictor import predict_fn
from .. import config

NBITS = config.nbits
MARKING_CHAR = config.marking_char

#for code in {1..256}; do printf "\e[38;5;${code}m"$code"\e[0m";echo; done
#for i = 1, 32 do COLORS[i] = "\27[38;5;"..(8*i-7).."m" end
#printf "\e[30;1mTesting color\e[0m"
#for i in range(25,50): print(f"\033[{i};1mTesting color {i}\033[0m")

class Show():
    COLORS = {
        "console": [
            "\033[48;1m", #grey
            "\033[34;1m", #blue first since often assayed
            "\033[31;1m", #red
            "\033[33;1m", #yellow
            "\033[32;1m", #green
            "\033[35;1m", #pink
            "\033[36;1m", #turquoise
            "\033[41;37;1m", #red back
            "\033[42;37;1m", #green back
            "\033[43;37;1m", #yellow back
            "\033[44;37;1m", #blue back
            "\033[45;37;1m" #turquoise back
        ],
        "html": [
            "<span style='color:grey'>",
            "<span style='color:blue'>",
            "<span style='color:red'>",
            "<span style='color:yellow'>",
            "<span style='color:green'>",
            "<span style='color:pink'>",
            "<span style='background-color:red;color:white'>",
            "<span style='background-color:green; color:white'>",
            "<span style='background-color:blue; color:white'>",
            "<span style='background-color:turquoise; color:white'>"
        ],
        "markdown": [""] * 12
    }

    CLOSE_COLOR = {
        "console": "\033[0m",
        "html": "</span>",
        "markdown": ""
    }

    BR = {
        "console": "\n",
        "html": "<br/>",
        "markdown": "\n"
    }

    N_COLORS = len(COLORS["console"])

    def __init__(self, format="markdown"):
        self.format = format
        self.col = Show.COLORS[format]
        self.close =  Show.CLOSE_COLOR[format]
        self.nl =  Show.BR[format]

    def example(self, dataset: Data4th, model: SmtagModel = None):
        out = ""
        rand_i = math.floor(len(dataset) * random())
        item = dataset[rand_i]
        nf_output = item.output.size(1)
        if model is not None:
            y, y_hat, loss = predict_fn(model, item, eval=True)
        out+= "\n__Expected:__" + "({})".format(item.provenance.strip()) + self.nl + self.nl
        out += self.print_pretty_color(item.target_class, nf_output, item.text) + self.nl + self.nl
        out += self.print_pretty(item.output) + self.nl + self.nl
        if model is not None:
            out += "__Predicted:__" + self.nl + self.nl
            out += self.print_pretty_color(y_hat.argmax(1), nf_output, item.text) + self.nl + self.nl
            out += self.print_pretty(y_hat.softmax(1)) + self.nl + self.nl
        out += ""
        print(out)
        return out

    SYMBOLS = ["_",".",":","^","|"] # for bins 0 to 0.1, 0.11 to 0.2, 0.21 to 0.3, ..., 0.91 to 1

    def print_pretty(self, features: BxCxL):
        out = ""
        N = len(Show.SYMBOLS) # = 5
        B = features.size(0)
        C = features.size(1)
        L = features.size(2)
        assert B==1
        track = self.nl + self.nl
        for i in range(C):
            track += f"Tagging track {i}" + self.nl
            for j in range(L):
                k = min(N-1, math.floor(N*features[0, i, j])) # 0 -> 0; 0.2 -> 1; 0.4 -> 2; 0.6 -> 3; 0.8 -> 4; 1.0 -> 4
                track += Show.SYMBOLS[k]
            track += self.nl + self.nl
        out += track + self.nl + self.nl
        return out

    def print_pretty_color(self, features: BxL, nf_features: int, text: str):
        text = text.replace(config.marking_char, '#')
        colored_track = "    "
        B = features.size(0)
        assert B==1
        for code, c in zip(features[0], text):
            colored_track += f"{self.col[nf_features - 1 - int(code.item())]}{c}{self.close}"
        return colored_track
