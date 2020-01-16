# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import unittest
import torch
from test.smtagunittest import SmtagTestCase
from smtag.common.converter import ConverterNBITS, TString, StringList, HeterogenousWordLengthError, RepeatError, ConcatenatingTStringWithUnequalDepthError
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
        self.assertEqual([text], s1.words)

    def test_slice(self):
        the_cat = StringList(["The","cat"])
        the_cat_ts = TString(the_cat)
        slices_ts = the_cat_ts[1:3]
        slices = StringList([s[1:3] for s in the_cat])
        expected = TString(slices)
        self.assertEqual(slices.words, slices_ts.words)
        self.assertTensorEqual(expected.tensor, slices_ts.tensor)

    def test_empty_string(self):
        empty_string = ''
        empty_string_ts = TString(empty_string)
        expected_string_list = []
        self.assertEqual(expected_string_list, empty_string_ts.words)
        self.assertEqual(empty_string_ts.tensor.dim(), 0)


    def test_concat_1(self):
        a = StringList(["the "])
        b = StringList(["cat"])
        ab = a + b
        t_ab = TString(a) + TString(b)
        print(ab.words)
        print(t_ab.words)
        self.assertEqual(ab.words, t_ab.words)
        self.assertTensorEqual(TString(ab).tensor, t_ab.tensor)

    def test_concat_2(self):
        a = StringList(["the ", "the "])
        b = StringList(["cat", "dog"])
        ab = a + b
        t_ab = TString(a) + TString(b)
        self.assertEqual(ab.words, t_ab.words)
        self.assertTensorEqual(TString(ab).tensor, t_ab.tensor)
        self.assertEqual(len(ab), 7)
        self.assertEqual(len(t_ab), 7)
        self.assertEqual(ab.depth, 2)
        self.assertEqual(t_ab.depth, 2)

    def test_concat_3(self):
        a = "the "
        b = ""
        ab = StringList([a+b])
        t_ab = TString(a) + TString(b)
        self.assertEqual(ab.words, t_ab.words)
        self.assertTensorEqual(TString(ab).tensor, t_ab.tensor)

    def test_concat_4(self):
        a = ""
        b = "cat"
        ab = StringList([a+b])
        t_ab = TString(a) + TString(b)
        self.assertEqual(ab.words, t_ab.words)
        self.assertTensorEqual(TString(ab).tensor, t_ab.tensor)

    def test_homogeneity(self):
        # StringList should have words of same length otherwise cannot be stacked into same tensor
        with self.assertRaises(HeterogenousWordLengthError):
            StringList(["the ", "a "])

    def test_concat_5(self):
        a = StringList(["the "])
        b = StringList(["cat", "dog"])
        with self.assertRaises(ConcatenatingTStringWithUnequalDepthError):
            TString(a) + TString(b)

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

    def test_repeat_2(self):
        c = "a"
        c10 = c * 10
        s = TString(StringList([c,c]))
        s10 = s.repeat(10)
        t10 = s.toTensor().repeat(1,1,10)
        print(s10.toStringList())
        self.assertTensorEqual(t10, s10.toTensor())
        self.assertListEqual([c10, c10], s10.words)

    def test_repeat_3(self):
        s = TString("a")
        with self.assertRaises(RepeatError):
            s.repeat(0)

if __name__ == '__main__':
    unittest.main()

