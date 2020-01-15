# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import unittest
import torch
from test.smtagunittest import SmtagTestCase
from smtag.common.converter import ConverterNBITS, TString, HeterogenousWordLengthError
from smtag.common.utils import timer
from string import ascii_letters
from random import choice


from smtag import config

NBITS = config.nbits
N_RANDOM_LETTERS = 10000
RANDOM_LETTERS = [choice(ascii_letters) for _ in range(N_RANDOM_LETTERS)]


class ConverterTest(SmtagTestCase):
    def setUp(self):
        self.input_string = u"😎😇"
        code = 0x03B1 #GREEK SMALL LETTER ALPHA Unicode: U+03B1, UTF-8: CE B1
        self.single_character = chr(code) # python 2: unichr(code)
        bits = [code >> i & 1 for i in range(NBITS)]
        self.tensor = torch.Tensor([int(b) for b in bits]).resize_(1,NBITS,1)

    def test_encode_string_into_tensor(self):
        converted = ConverterNBITS().encode(self.single_character)
        self.assertTensorEqual(self.tensor, converted)

    def test_decode_tensor_into_string(self):
        self.assertEqual(self.single_character, ConverterNBITS().decode(self.tensor))

    def test_lossless_encode_decode(self):
        self.assertEqual(self.input_string, ConverterNBITS().decode(ConverterNBITS().encode(self.input_string)))

    def test_lossless_decode_encode(self):
        self.assertTensorEqual(self.tensor, ConverterNBITS().encode(ConverterNBITS().decode(self.tensor)))

    @timer
    def test_timing(self):
        c = ConverterNBITS()
        for i in range(N_RANDOM_LETTERS):
            c.encode(RANDOM_LETTERS[i])

class TStringTest(SmtagTestCase):

    def test_lossless_decode_encode(self):
        text = "hallo"
        s1 = TString(text)
        tensor = s1.toTensor()
        s2 = TString(tensor)
        self.assertTensorEqual(s1.toTensor(), s2.toTensor())
        self.assertEqual(s1.toTensor().size(), s2.toTensor().size())
        self.assertEqual(len(s1), len(s1))
        self.assertNotEqual(len(s2), 0)
        self.assertEqual([text], s1.toStringList())

    def test_slice(self):
        the_cat = ["The","cat"]
        the_cat_ts = TString(the_cat)
        slices_ts = the_cat_ts[1:3]
        slices = [s[1:3] for s in the_cat]
        expected = TString(slices)
        self.assertEqual(slices, slices_ts.toStringList())
        self.assertTensorEqual(expected.tensor, slices_ts.tensor)

    def test_empty_string(self):
        empty_string = ''
        empty_string_ts = TString(empty_string)
        expected_string_list = []
        self.assertEqual(expected_string_list, empty_string_ts.toStringList())
        self.assertEqual(empty_string_ts.tensor.dim(), 0)

    def test_concat_1(self):
        a = ["the ", "the "]
        b = ["cat", "dog"]
        ab = [first + second for first, second in zip(a, b)]
        t_ab = TString(a) + TString(b)
        self.assertEqual(ab, t_ab.toStringList())
        self.assertTensorEqual(TString(ab).tensor, t_ab.tensor)

    def test_concat_2(self):
        a = "the "
        b = ""
        ab = [a+b]
        t_ab = TString(a) + TString(b)
        self.assertEqual(ab, t_ab.toStringList())
        self.assertTensorEqual(TString(ab).tensor, t_ab.tensor)

    def test_concat_3(self):
        a = ""
        b = "cat"
        ab = [a+b]
        t_ab = TString(a) + TString(b)
        self.assertEqual(ab, t_ab.toStringList())
        self.assertTensorEqual(TString(ab).tensor, t_ab.tensor)

    def test_homogeneity(self):
        a = ["the ", "a "] # first word StringList should have words of same length otherwise cannot be stacked into same tensor
        with self.assertRaises(HeterogenousWordLengthError):
            TString(a)

    def test_len(self):

        x = TString("1234567890")
        l1 = len(x)
        l2 = len("1234567890")
        self.assertEqual(l1, l2)

    def test_repeat_1(self):
        c = "a"
        c10 = c * 10
        s = TString(c)
        s10 = s.repeat(10)
        t10 = s.toTensor().repeat(1,1,10)
        self.assertTensorEqual(t10, s10.toTensor())


if __name__ == '__main__':
    unittest.main()

