# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import numpy as np
import argparse
import torch
from .utils import timer

class Converter():
    """
    Conversion operations between unicode strings and tensors.
    """

    @staticmethod
    def t_encode(input_string, dtype=torch.float):
        """
        Static method that encodes an input string into a 3D tensor.
        Args
            input_string (str): string to convert
        Returns
            (torch.Tensor): 3D tensor 1 x 32 x L, (1 example x 32 bits x L characters) representing characters as 32 features
        """
        
        L = len(input_string)
        t = torch.zeros(1, 32, L, dtype=dtype)
        for i in range(L):
            code = ord(input_string[i])
            # the integer is first represented as binary in a padded string with bin(code) and [2:] to remove "0b"
            # to fill the tensor, the bits need to be read from right to left, so string or array needs to be reversed
            # the array is then reversed using [::-1] or the string with reversed()
            # in this way the bits from right to left populate the final Tensor (column) from left (top) to right (bottom)
            # bits = torch.Tensor([int(b) for b in "{0:032b}".format(code)][::-1]) # slower!! 2.425s for 1E5 conversions; thank you: https://stackoverflow.com/questions/10321978/integer-to-bitfield-as-a-list
            # bits = torch.Tensor([1 if b=='1' else 0 for b in "{0:032b}".format(code)][::-1]) # faster: 1.721s
            # bits = torch.Tensor([1 if b=='1' else 0 for b in f"{code:032b}"][::-1]) # elegant but 1.7s and only python 3.6
            #bits = torch.Tensor([1 if b=='1' else 0 for b in "%32s" % bin(code)[2:]][::-1]) # even faster 1.653s with % formatting
            bits = torch.Tensor([1 if b=='1' else 0 for b in reversed("%32s" % bin(code)[2:])]) # more elegant
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

        L = t.size(2)
        str = ""
        for i in range(L):
            code = 0
            for j in range(31):
                bit = int(t[0, j, i])
                code += bit*(2**j)
            str += chr(code) #python 2: unichr()
        return str

class TString: # (str) or (torch.Tensor)?
    '''
    Class to represent strings simultaneously as Tensor and as str. String is encoded into a 3D (1 example x 32 bits x L characters) Tensor.

    Args:
        x: either a string, in in which case it is converted into the corresonding Tensor;
        or a Tensor, in which case it does not need conversion but needs to be 3 dim with size(1)==NBITS (32).
        If no argument is provided, TString is initialized with an empty string.

    Methods:
        __str__(): string representation of TString
        __len__(): length with len(TString) and returns int
        __add__(TString): concatenates TString and returns a TString; allow operation like tstring_1 + tstring_2 
        __getitem(i): gets the i-th element of the string and of the underlying tensor and returns a TString; allows to slice with tstring[start:stop]
        repeat(N): repeats the TString N time
        toTensor(): returns the torch.Tensor representation of the encoded string
        all the remainint methods from torch.Tensor
    '''

    def __init__(self, x='', dtype=torch.float):
        super(TString, self).__init__()
        self.dtype = dtype
        self.t = torch.zeros([], dtype=self.dtype) # empty tensor
        if isinstance(x, str):
            self.t = Converter.t_encode(x, dtype=self.dtype)
            self.s = x
        elif isinstance(x, torch.Tensor):
            assert(x.dim() == 3 and x.size(1) == 32)
            self.t = x
            self.s = Converter.t_decode(x)

    def __str__(self):
        return self.s

    def __len__(self):
        return len(self.s)

    def __add__(self, x): # overwrites tensor adding into tensor concatenation like strings
        if len(x) == 0:
            return self # or should it return a cloned self?
        elif len(self.s) == 0:
            return x # or should it return a cloned x?
        else:
            concatenated = TString(dtype=self.dtype)
            concatenated.t = torch.cat((self.toTensor(), x.toTensor()), 2)
            concatenated.s = str(self) + str(x)
            return concatenated

    def __getitem__(self, i):
        if len(self.s) == 0:
            return TString()
        else:
            item = TString(dtype=self.dtype)
            item.s = self.s[i]
            item.t = self.t[ : , : , i]
            return item

    def repeat(self, N):
        repeated = TString(dtype=self.dtype)
        repeated.t = self.t.repeat(1, 1, N) # WARNING: if N == 0, returned tensor is 2D !!!
        repeated.s = self.s * N
        return repeated

    def toTensor(self):
        return self.t

    def __getattr__(self, attr): # class composition with tensor.Torch
        return getattr(self.t, attr)




if __name__ == "__main__":
    # more systematic tests in test.test_converter
    parser = argparse.ArgumentParser( description="Encode decode string into binary tensors" )
    parser.add_argument('input_string', nargs='?', default= "this is so ☾😎 😎 L ‼️", help="The string to convert")
    args = parser.parse_args()
    input_string = args.input_string#.encod('utf-8')
    print("the decoded of encoded:", Converter.t_decode(Converter.t_encode(input_string)))
    
    
