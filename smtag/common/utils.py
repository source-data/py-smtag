# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import re
from xml.sax.saxutils import escape
from collections import namedtuple
from contextlib import contextmanager
from xml.etree.ElementTree import Element, fromstring, tostring
import os
import time


def xml_escape(s):
    return escape(s)


def cleanup(text):
    text = re.sub('[\r\n\t]', ' ', text)
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[–—‐−]', '-', text) # controversial!!!
    return text

def innertext(element: Element) -> str:
    return "".join([t for t in element.itertext()])

def special_innertext(element:Element, tag_list = ['sd-panel', 'label', 'b']) -> str:
    def add_tail_space(element: Element):
        for e in element:
            if e.tag in tag_list:
                if e.tail is None: 
                    e.tail = ' '
                elif e.tail[0] != ' ':
                    e.tail = ' ' + e.tail
            add_tail_space(e)

    def remove_double_spaces(element: Element):
        s = tostring(element, encoding = "unicode")
        replaced = re.sub(r' ((?:<[^>]+>)+) ', r' \1', s)
        xml = fromstring(replaced)
        return xml

    add_tail_space(element)
    no_double_space = remove_double_spaces(element)
    inner_text = innertext(no_double_space)
    return inner_text, no_double_space

Token = namedtuple('Token', ['text', 'start', 'stop', 'length', 'left_spacer']) # should be a proper object with __len__ method

def tokenize(s):
    #patterns derived from python nltk library http://www.nltk.org/_modules/nltk/tokenize/punkt.html#PunktLanguageVars.word_tokenize

    re_word_start = r"[^\./\(\"\`{\[:;&\#\*@\)}\]\-–—‐−,]" #added . and / and dashes
    re_non_word_chars = r"(?:[\.?!)\";}\]\*:@\'\({\[,])" # added . and +
    re_multi_char_punct = r"(?:\-–—‐−{2,}|\.{2,}|(?:\.\s){2,}\.)" # added dashes
    re_genotype = r"(?:[+\-–—‐−]/[+\-–—‐−])"
    re_hyphenated = r"(?<!\s[\dΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩωA-Za-z])(?:[\-–—‐−][^\dΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω])" # try to modify first term to (?<!\s[^\dαβγδεζηθικλμνξπρστυφχψωA-Z]) to include single capital or single letter with A-Za-z] IL-β or γ-actin vs GFP-P53 or 12-330-3244 or Creb-1
    re_word_tokenize_fmt = r'''(
        %(MultiChar)s
        |
        %(Genotype)s
        |
        (?=%(WordStart)s)\S+?  # Accept word characters until end is found; the non-greedy operator ? is essential
        (?= # Sequences marking a word's end
            \s|                                         # White-space
            $|                                          # End-of-string
            %(NonWord)s|%(Genotype)s|%(Hyphenated)s  # Punctuation and Hyphenated

        )
        |
        \S
    )'''

    tokenizer = re.compile(re_word_tokenize_fmt % {
                    'NonWord': re_non_word_chars,
                    'MultiChar': re_multi_char_punct,
                    'WordStart': re_word_start,
                    'Genotype': re_genotype,
                    'Hyphenated': re_hyphenated
                    }, re.VERBOSE)

    matches = tokenizer.finditer(s)
    token_list = []
    left_spacer = ''
    last_stop = 0
    start_index = [] # indexing by start position of the token
    stop_index = [] # indexing by stop position of the token
    for m in matches:
        start = m.start()
        stop = m.end()
        length = stop - start
        text = m.group(0)
        left_spacer = s[last_stop:start]
        t = Token(text=text, start=start, stop=stop, length=length, left_spacer=left_spacer)
        token_list.append(t)
        start_index.append(start)
        stop_index.append(stop)
        last_stop = stop

    return {'token_list':token_list, 'start_index':start_index, 'stop_index':stop_index}

@contextmanager
def cd(newdir):
    '''
    From: https://stackoverflow.com/questions/431684/how-do-i-change-directory-cd-in-python/24176022#24176022
    '''

    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def timer(f):
    '''
    A decorator to print the execution time of a method.
    Usage:
        @timer
        def some_function_to_profile(x, y, z):
    '''
    def t(*args, **kwargs):
        start_time = time.time()
        output = f(*args, **kwargs)
        end_time = time.time()
        delta_t = end_time - start_time
        print("\nExec time for '{}': {:.3f}s".format(f.__name__, delta_t))
        return output
    return t
