# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import argparse
import torch
from .utils import timer
from .. import config

NBITS = config.nbits


class Converter():
    """
    Conversion operations between unicode strings and tensors.
    """

    @staticmethod
    def t_encode(input_string: str, dtype:torch.dtype=torch.float) -> torch.Tensor:
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
            t = torch.zeros(1, NBITS, L, dtype=dtype)
            for i in range(L):
                code = ord(input_string[i])
                bits = torch.Tensor([code >> i & 1 for i in range(NBITS)]) # the beloved bitwise operation, not faster but not slower: 1.902 sec :-)
                t[0, : , i] = bits
        return t

    @staticmethod
    def t_decode(t: torch.Tensor) -> str:
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

class TString: # (str) or (torch.Tensor)?
    '''
    Class to represent strings simultaneously as Tensor and as str. String is encoded into a 3D (1 example x NBITS bits x L characters) Tensor.

    Args:
        x: either a string, in in which case it is converted into the corresonding Tensor;
        or a Tensor, in which case it does not need conversion but needs to be 3 dim with size(1)==NBITS (NBITS).
        If no argument is provided, TString is initialized with an empty string.

    Methods:
        __str__(): string representation of TString
        __len__(): length with len(TString) and returns int
        __add__(TString): concatenates TString and returns a TString; allow operation like tstring_1 + tstring_2
        __getitem(i): gets the i-th element of the string and of the underlying tensor and returns a TString; allows to slice with tstring[start:stop]
        repeat(N): repeats the TString N time
        toTensor(): returns the 3D (1 x NBITS x L) torch.Tensor representation of the encoded string
        all the remaining methods from torch.Tensor
    '''

    def __init__(self, x:str='', dtype:torch.dtype=torch.float):
        #super(TString, self).__init__()
        self.dtype = dtype
        self.t = torch.zeros([], dtype=self.dtype) # empty tensor
        if isinstance(x, str):
            self.t = Converter.t_encode(x, dtype=self.dtype)
            self.s = x
        elif isinstance(x, torch.Tensor):
            assert x.dim() == 3 and x.size(1) == NBITS
            self.t = x
            self.s = Converter.t_decode(x)

    def __str__(self) -> str:
        return self.s

    def __len__(self) -> int:
        return len(self.s)

    def __add__(self, x: 'TString') -> 'TString': # using string as type because python 3.6 does not allow using class as type before it is defined
        # overwrites tensor adding operator to make it a tensor concatenation like for strings
        if len(x) == 0:
            return self # or should it return a cloned self?
        elif len(self.s) == 0:
            return x # or should it return a cloned x?
        else:
            concatenated = TString(dtype=self.dtype)
            concatenated.t = torch.cat((self.toTensor(), x.toTensor()), 2)
            concatenated.s = str(self) + str(x)
            return concatenated

    def __getitem__(self, i: int) -> 'TString':
        if len(self.s) == 0:
            return TString()
        else:
            item = TString(dtype=self.dtype)
            item.s = self.s[i]
            item.t = self.t[ : , : , i]
            return item

    def repeat(self, N: int) -> 'TString':
        repeated = TString(dtype=self.dtype)
        repeated.t = self.t.repeat(1, 1, N) # WARNING: if N == 0, returned tensor is 2D !!!
        repeated.s = self.s * N
        return repeated

    def toTensor(self) -> torch.Tensor:
        return self.t

    def __getattr__(self, attr: str): # class composition with tensor.Torch
        return getattr(self.t, attr)


def self_test(input_string: str):
    decode_encoded = Converter.t_decode(Converter.t_encode(input_string))
    assert input_string == decode_encoded, f"{input_string}<>{decode_encoded}"
    print("This seems to work!")

def main():
    # more systematic tests in test.test_converter
    parser = argparse.ArgumentParser( description="Encode decode string into binary tensors" )
    parser.add_argument('input_string', nargs='?', default= "αβγ∂this is so ☾😎 😎 L ‼️" + u'\uE000', help="The string to convert")
    args = parser.parse_args()
    input_string = args.input_string#.encode('utf-8')
    print(f"NBITS={NBITS}")
    print("the decoded of the encoded:", Converter.t_decode(Converter.t_encode(input_string)))
    self_test(input_string)

if __name__ == "__main__":
    main()
