# -*- coding: utf-8 -*-
import re
from xml.sax.saxutils import escape
from collections import namedtuple

def xml_escape(s):
    return escape(s)

Token = namedtuple('Token', ['text', 'start', 'stop', 'left_spacer'])

def tokenize(s):
    #patterns derived from python nltk library http://www.nltk.org/_modules/nltk/tokenize/punkt.html#PunktLanguageVars.word_tokenize
    
    re_word_start = r"[^\./\(\"\`{\[:;&\#\*@\)}\]\-–—‐−,]" #added . and / and dashes
    re_non_word_chars = r"(?:[\.?!)\";}\]\*:@\'\({\[,])" # added . and + 
    re_multi_char_punct = r"(?:\-–—‐−{2,}|\.{2,}|(?:\.\s){2,}\.)" # added dashes
    re_genotype = r"(?:[+\-–—‐−]/[+\-–—‐−])"
    re_hyphenated = r"(?<=[^\dαβγδεζηθικλμνξπρστυφχψω])(?:[\-–—‐−][^\dαβγδεζηθικλμνξπρστυφχψω])" # IL-β or γ-actin vs GFP-P53 or 12-330-3244 or Creb-1

    re_word_tokenize_fmt = r'''(
        %(MultiChar)s
        |
        %(Genotype)s
        |
        (?=%(WordStart)s)\S+?  # Accept word characters until end is found
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
    for m in matches:
        start = m.start()
        stop = m.end()
        text = m.group(0)
        left_spacer = s[last_stop:start]
        t = Token(text=text, start=start, stop=stop, left_spacer=left_spacer)
        token_list.append(t)
        last_stop = stop

    return token_list

class TokenIter:
    '''
    Iterator over token lists that returns next token and either start or stop depending on parameter mode.
    '''

    def __init__(self, token_list, mode):
        self.token_list = iter(token_list)
        self.mode = mode
    def __iter__(self):
        return self
    def __next__(self):
        if self.mode == 'start':
           next_token = next(self.token_list)
           return (next_token.start, next_token.text)
        elif self.mode == 'stop':
           next_token = next(self.token_list)
           return (next_token.stop, next_token.text)


class StringIter:
    '''
    Iterator over string that returns next character and next position.
    '''
    def __init__(self, s):
        self.s = iter(s)
        self.range_L = range(len(s))
    def __iter__(self):
        return self
    def __next__(self):
        next_i = next(self.range_L)
        next_char = next(self.s)
        return (next_i, next_char)

class PositionIter:
    '''
    Iterates either on a string or, if token are available, on list of token. If it iterates on token, mode allows to select whether it is it start position, stop position or its text that is yielded
    The idea is to allow to go faster through text by jumping from token to token, provided the text has already been tokenized.
    
    Args:
        L (int): makes PositionIter a normal range(L)
        token (list): if not empty, PositionIter is a TokenIter which returns one field of each token
        mode ('start'|'stop'|'text'): the field that needs to be yielded at each step
    '''
    
    def __init__(self, s='', token=[], mode='start'):
        if token:
           self.it = TokenIter(token, mode)
        else:
           self.it = StringIter(s)
    def __iter__(self): 
           return self.it.__iter__()
    def __next__(self):
           return next(self.it)