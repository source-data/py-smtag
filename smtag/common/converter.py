# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import argparse
import torch
from .utils import timer
from .. import config

NBITS = config.nbits

class Converter():
    """
    Conversion operations between strings and tensors.
    """
    def __init__(self, dtype:torch.dtype=torch.float):
        self.dtype = dtype

    def encode(self, input_string:str) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, t: torch.Tensor) -> str:
        raise NotImplementedError

class ConverterNBITS(Converter):

    def __init__(self, dtype:torch.dtype=torch.float):
       super(ConverterNBITS, self).__init__(dtype)

    def encode(self, input_string: str) -> torch.Tensor:
        """
        Static method that encodes an input string into a 3D tensor.
        Args
            input_string (str): string to convert
        Returns
            (torch.Tensor): 3D tensor 1 x NBITS x L, (1 example x NBITS bits x L characters) representing characters as NBITS features
        """

        L = len(input_string)
        t = torch.Tensor(0)
        if L > 0:
            t = torch.zeros(1, NBITS, L, dtype=self.dtype)
            for i in range(L):
                code = ord(input_string[i])
                bits = torch.Tensor([code >> i & 1 for i in range(NBITS)]) # the beloved bitwise operation, not faster but not slower: 1.902 sec :-)
                t[0, : , i] = bits
        return t

    def decode(self, t: torch.Tensor) -> str:
        """
        Static method that decodes a 3D tensor into a unicode string.
        Args:
            t (torch.Tensor): 3D tensor 1xNBITSxL (1 example x NBITS bits x L characters) representing characters as NBITS features
        Returns
            (str): resulting string
        """

        L = t.size(2)
        str = ""
        for i in range(L):
            code = 0
            for j in range(NBITS):
                bit = t[0, j, i]
                try:
                    bit = int(bit)
                except Exception as e:
                    print(f"{e} in converter module.")
                    print(e)
                    bit = 0
                code += bit*(2**j)
            try:
                str += chr(code) #python 2: unichr()
            except ValueError:
                str += '?'
            except OverflowError:
                print("Error: code too large", code)
                print(type(code))
                print(bin(code))
                print(t[0, : , i].view(-1))
                str += '?'
        return str

class TString:
    '''
    Class to represent strings simultaneously as Tensor and as str. String is encoded into a 3D Tensor.
    The number of feature is NBITS.

    Args:
        x: either a string, in in which case it is converted into the corresonding Tensor;
        or a Tensor, in which case it does not need conversion but needs to be 3 dim .
        If no argument is provided, TString is initialized with an empty string.

    Methods:
        __str__(): string representation of TString
        __len__(): length with len(TString) and returns int
        __add__(TString): concatenates TString and returns a TString; allow operation like tstring_1 + tstring_2
        __getitem(i): gets the i-th element of the string and of the underlying tensor and returns a TString; allows to slice with tstring[start:stop]
        repeat(N): repeats the TString N time
        tensor: returns the 3D (1 x NBITS x L) torch.Tensor representation of the encoded string
        all the remaining methods from torch.Tensor
    '''

    def __init__(self, x='', dtype:torch.dtype=torch.float):
        #super(TString, self).__init__()
        self.dtype = dtype
        self._t = torch.zeros([], dtype=self.dtype) # empty tensor
        self._s = ''
        if isinstance(x, str):
            converter = ConverterNBITS(dtype=self.dtype)
            self._t = converter.encode(x)
            self._s = x
        elif isinstance(x, torch.Tensor):
            assert x.dim() == 3 and x.size(1) == NBITS
            converter = ConverterNBITS(dtype=self.dtype)
            self._t = x
            self._s = converter.decode(x)

    def __str__(self) -> str:
        return self._s

    def __len__(self) -> int:
        return len(self._s)

    def __add__(self, x: 'TString') -> 'TString': # using string as type hint because python 3.6 does not allow using class as type before it is defined
        # overwrites tensor adding operator to make it a tensor concatenation like for strings
        if len(x) == 0:
            return self # or should it return a cloned self?
        elif len(self._s) == 0:
            return x # or should it return a cloned x?
        else:
            concatenated = TString(dtype=self.dtype)
            concatenated._t = torch.cat((self.tensor, x.tensor), 2)
            concatenated._s = str(self) + str(x)
            return concatenated

    def __getitem__(self, i: int) -> 'TString':
        if len(self._s) == 0:
            return TString()
        else:
            item = TString(dtype=self.dtype)
            item._s = self._s[i]
            item._t = self._t[ : , : , i]
            return item

    def repeat(self, N: int) -> 'TString':
        repeated = TString(dtype=self.dtype)
        repeated._t = self._t.repeat(1, 1, N) # WARNING: if N == 0, returned tensor is 2D !!!
        repeated._s = self._s * N
        return repeated

    @property
    def tensor(self) -> torch.Tensor:
        return self._t

    def toTensor(self) -> torch.Tensor: # legacy method
        return self._t

    def __getattr__(self, attr: str): # class composition with tensor.Torch
        return getattr(self._t, attr)


def self_test(input_string: str):
    encoded = ConverterNBITS().encode(input_string)
    decode_encoded = ConverterNBITS().decode(encoded)
    print("the decoded of the encoded:", str(TString(TString(input_string).tensor)))
    assert input_string == decode_encoded, f"{input_string}<>{decode_encoded}"

def main():
    # more systematic tests in test.test_converter
    parser = config.create_argument_parser_with_defaults(description="Encode decode string into binary tensors")
    parser.add_argument('input_string', nargs='?', default= "αβγ∂this is so ☾😎 😎 L ‼️" + u'\uE000', help="The string to convert")
    args = parser.parse_args()
    input_string = args.input_string
    self_test(input_string)

if __name__ == "__main__":
    main()
