# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import numpy as np
import argparse
import torch

class Converter():
    """
    Conversion operations between unicode strings and tensors.
    """
    
    @staticmethod
    def t_encode(input_string):
        """
        Static method that encodes an input string into a 4D tensor.
        Args
            input_string (str): string to convert
        Returns
            (torch.Tensor): 3D tensor 1x32x1xL (1 example x 32 bits x L characters) representing characters as 32 features
        """
        
        L = len(input_string)
        t = torch.zeros(1,32,L)
        for i in range(L):
            code = ord(input_string[i])
            # the integer is first represented as binary in a string
            # the bits are read from left to right to fill the tensor
            # the tensor is then inverted using [::-1]
            # in this way the bits from right to left populate the final Tensor (column) from left (top) to right (bottom)
            bits = torch.Tensor([int(b) for b in "{0:032b}".format(code)][::-1])
            t[0, : , i] = bits
        return t

    @staticmethod
    def t_decode(t):
        """
        Static method that decodes a 3D tensor into a unicode string.
        Args:
            t (torch.Tensor): 3D tensor 1x32xL (1 example x 32 bits x L characters) representing characters as 32 features
        Returns
            (str): resulting string
        """
        #tensor is 3D
        L = t.size(2)
        str = ""
        for i in range(L):
            code = 0
            for j in range(31):
                bit = int(t[0, j, i])
                code += bit*(2**j)
            str += chr(code) #python 2: unichr()
        return str

class TString(object): # (str) or (torch.Tensor)?
    '''
    Composition between torch tensor and string such that both representation coexist. A string is converted into a 3D 1 x 32 x L Tensor and vice versa.

    Args:
        x: either a string, in in which case it is converted into the corresonding Tensor, or a Tensor, in which case it is converted into a string.

    Methods:
        all methods from torch.Tensor
        __str__(): allows to print the TString
        __len__(): provide length with len(TString)
    '''
    def __init__(self, x):
        if isinstance(x, str):
            self.s = x
            self.t = Converter.t_encode(x)
        elif isinstance(x, torch.Tensor):
            self.s = Converter.t_decode(x)
            self.t = x

    def __str__(self):
        return self.s

    def __len__(self):
        return len(self.s)

    def __getitem__(self, key):
        return self.s[key]

    def __getattr__(self, attr):
        return getattr(self.t, attr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="Encode decode string into binary tensors" )
    parser.add_argument('input_string', nargs='?', default= "this is so ☾😎 😎 L ‼️", help="The string to convert")
    args = parser.parse_args()
    input_string = args.input_string#.encod('utf-8')
    print("the decoded of encoded:", Converter.t_decode(Converter.t_encode(input_string)))
    
    
