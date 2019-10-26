# -*- coding: utf-8 -*-
#T. Lemberger, 2018

#from abc import ABC
import torch
import json
from xml.etree.ElementTree import fromstring
from copy import deepcopy
from typing import List
from collections import OrderedDict
from ..common.utils import xml_escape, timer
from ..common.mapper import Catalogue, Boundary
from ..predict.decode import Decoder


class AbstractElementSerializer(object):

    @staticmethod
    def make_element(inner_text):
        pass

    @staticmethod
    def mark_boundary(action):
        pass

    @staticmethod
    def map(tag, on_features, concept):
        pass

class XMLElementSerializer(AbstractElementSerializer):

    @staticmethod
    def make_element(tag, concepts, inner_text, scores):
        xml_string = ''
        if inner_text:
            attribute_list= {}
            attribute_score = {}
            for group in concepts:
                concept = concepts[group]
                if concept != Catalogue.UNTAGGED:
                    score = scores[group]
                    attribute, value = concept.for_serialization
                    attribute_score[attribute] = score
                    attribute_list[attribute] = value
            xml_attributes = ' '.join(['{}="{}"'.format(a, attribute_list[a]) for a in attribute_list])
            xml_scores = ' '.join(['{}_score="{}"'.format(a, str(int(100*attribute_score[a]))) for a in attribute_score])
            xml_string = "<{} {} {}>{}</{}>".format(tag, xml_attributes, xml_scores, inner_text, tag)
        return xml_string # because both HTML and XML handled, working with strings is easier than return ET.tostring(xml_string)

    @staticmethod
    def mark_boundary(action): # in the future, it will accept a specific boundary but now only PANEL
        if action == 'open':
            return '<{}>'.format(Catalogue.PANEL_START.for_serialization) # discrepancy: here via Catalogue, whereas via param tag for elements
        elif action == 'close':
            return '</{}>'.format(Catalogue.PANEL_START.for_serialization)


class HTMLElementSerializer(AbstractElementSerializer):

    @staticmethod
    def make_element(tag, concepts, inner_text, scores):
        html_string = ''
        if inner_text:
            attribute_list = {}
            attribute_score = {}
            for group in concepts:
                concept = concepts[group]
                if concept != Catalogue.UNTAGGED:
                    score = scores[group]
                    attribute, value = concept.for_serialization
                    attribute_list[attribute] = value
                    attribute_score[attribute] = score
            html_classes = ' '.join([a + "_" + attribute_list[a] for a in attribute_list])
            score_classes = ' '.join([a + "_score_" + str(int(100*attribute_score[a])) for a in attribute_score])
            html_string = "<span class=\"{} {} {}\">{}</span>".format(tag, html_classes, score_classes, inner_text)
        return html_string

    @staticmethod
    def mark_boundary(action): # in the future, it will accept a specific boundary but now only PANEL
        if action == 'open':
            return '<span class="{}">'.format(Catalogue.PANEL_START.for_serialization)
        elif action == 'close':
            return '</span>'


class AbstractTagger:

    def __init__(self, tag):
        self.tag = tag
        self.serialized_examples = []

    def serialize_element(self, current_concepts, inner_text, current_scores):
        raise NotImplementedError

    def serialize_boundary(self, action):
        raise NotImplementedError

    @property
    def opening_tag(self):
        raise NotImplementedError

    @property
    def closing_tag(self):
        raise NotImplementedError


    def panel_segmentation(self, decoded: Decoder) -> List:
        panels = []
        indices = [i for i, c in enumerate(decoded.char_level_concepts['panels']) if c == Catalogue.PANEL_STOP]
        token_list = deepcopy(decoded.token_list)
        for i in indices:
            panel = []
            stop = 0
            while stop < i:
                t = token_list.pop(0)
                panel.append(t)
                stop = t.stop
            panels.append(panel)
        rest= [t for t in token_list]
        if rest:
            panels.append(rest)
        return panels

    def serialize(self, decoded: Decoder) -> str:
        ml_string = ""
        inner_text = ""
        pos = 0
        num_open_elements = 0
        current_concepts = OrderedDict([(g, Catalogue.UNTAGGED) for g in decoded.semantic_groups]) # initialize with UNTAGGED?
        need_to_open = OrderedDict([(g, False) for g in decoded.semantic_groups])
        need_to_close = OrderedDict([(g, False) for g in decoded.semantic_groups])
        need_to_open_any = False
        need_to_close_any = False
        current_scores = {g: 0 for g in decoded.semantic_groups}

        if 'panels' in decoded.semantic_groups:
            panels = self.panel_segmentation(decoded)
            open_tag = self.opening_tag
            closing_tag = self.closing_tag
        else:
            panels = [decoded.token_list]
            open_tag = ''
            closing_tag = ''
        for panel in panels:
            if ml_string: # if in the middle of a multi-panel legend, 
                ml_string += panel[0].left_spacer # need first to add the spacer of first token of next panel
                ml_string += closing_tag # and close panel
            ml_string += open_tag
            for count, token in enumerate(panel):
                text = xml_escape(token.text)
                left_spacer = token.left_spacer if count > 0 else ""
                for group in decoded.semantic_groups: # scan across feature groups the features that need to be opened
                    concept = decoded.concepts[group][pos]
                    if concept != Catalogue.UNTAGGED and concept != current_concepts[group]: # a new concept
                        need_to_open[group] = concept
                        need_to_open_any = True
                        if current_concepts[group] == Catalogue.UNTAGGED:
                            num_open_elements += 1

                # print(f"2.inner_text: '{inner_text}', ml_string: '{ml_string}'"); import pdb; pdb.set_trace()

                if need_to_open_any:
                    need_to_open_any = False
                    tagged_string = self.serialize_element(current_concepts, inner_text, current_scores)
                    ml_string += tagged_string + left_spacer # add the tagged alement to the nascent markup string
                    inner_text = text

                    # print(f"3.inner_text: '{inner_text}', ml_string: '{ml_string}'"); import pdb; pdb.set_trace()

                    for group in decoded.semantic_groups:
                        concept = decoded.concepts[group][pos]
                        current_scores[group] = decoded.scores[group][pos].item()
                        if need_to_open[group]: # CHANGED
                            current_concepts[group] = concept
                            need_to_open[group] = False
                        elif current_concepts[group] != Catalogue.UNTAGGED and concept == Catalogue.UNTAGGED:
                            num_open_elements -= 1
 
                    # print(f"4.inner_text: '{inner_text}', ml_string: '{ml_string}'"); import pdb; pdb.set_trace()

                else:
                    for group in decoded.semantic_groups:
                        concept = decoded.concepts[group][pos]
                        if current_concepts[group] != Catalogue.UNTAGGED and concept == Catalogue.UNTAGGED:
                            need_to_close[group] = True
                            need_to_close_any = True
                            num_open_elements -= 1

                    # print(f"5.inner_text: '{inner_text}', ml_string: '{ml_string}'"); import pdb; pdb.set_trace()

                    if need_to_close_any:
                        need_to_close_any = False
                        tagged_string = self.serialize_element(current_concepts, inner_text, current_scores)
                        ml_string += tagged_string + left_spacer
                        inner_text = ''
                        if num_open_elements > 0:
                            inner_text = text
                        else:
                            ml_string += text

                        # print(f"6.inner_text: '{inner_text}', ml_string: '{ml_string}'"); import pdb; pdb.set_trace()

                        for group in decoded.semantic_groups:
                            if need_to_close[group]:
                                need_to_close[group] = False
                                current_concepts[group] = Catalogue.UNTAGGED
                                current_scores[group] = 0
                    else:
                        if num_open_elements > 0:
                            inner_text += left_spacer + text
                        else:
                            ml_string += left_spacer + text
                pos += 1

            if num_open_elements > 0:
                tagged_string = self.serialize_element(current_concepts, inner_text, current_scores)

                # print(f"7.inner_text: '{inner_text}', ml_string: '{ml_string}'"); import pdb; pdb.set_trace()

        ml_string += closing_tag # hmm in principle left spacer of first token of next panel should go in here
        #phew!
        return ml_string

class XMLTagger(AbstractTagger):

    def __init__(self, tag):
        super(XMLTagger, self).__init__(tag)

    @property
    def opening_tag(self):
        return '<sd-panel>'

    @property
    def closing_tag(self):
        return '</sd-panel>'

    def serialize_element(self, on_features, inner_text, current_scores):
        return XMLElementSerializer.make_element(self.tag, on_features, inner_text, current_scores)

    def serialize_boundary(self, action): # preliminary naive implementation...
        return XMLElementSerializer.mark_boundary(action)

    def serialize(self, decoded_pred):
        xml_string = super(XMLTagger, self).serialize(decoded_pred)
        # need to provide valid xml
        return "<smtag>{}</smtag>".format(xml_string)

class HTMLTagger(AbstractTagger):

    def __init__(self, tag):
        super(HTMLTagger, self).__init__(tag)

    @property
    def opening_tag(self):
        return '<li class="sd-panel">'

    @property
    def closing_tag(self):
        return '</li>'

    def serialize_element(self, on_features, inner_text, current_scores):
        return HTMLElementSerializer.make_element(self.tag, on_features, inner_text, current_scores)

    def serialize_boundary(self, action):
        return HTMLElementSerializer.mark_boundary(action)

    def serialize(self, decoded_pred):
        html_string = super(HTMLTagger, self).serialize(decoded_pred)
        return "<ul>{}</ul>".format(html_string)

class JSONTagger(XMLTagger):

    def __init__(self, tag):
        super(JSONTagger, self).__init__(tag)

    def serialize(self, decoded_pred):
        xml_string = super(JSONTagger, self).serialize(decoded_pred)
        xml = fromstring(xml_string)
        j =  {
            'smtag': []
        }
        panels = xml.findall('sd-panel')
        if not panels:
            panels = [xml]
        for panel in panels:
            entities = []
            for e in panel.findall(self.tag):
                entity = e.attrib
                entity['text'] = e.text
                if entity not in entities:
                    entities.append(entity)
            j['smtag'].append({'entities': entities})
        js = json.dumps(j)
        return js

class Serializer():

    def __init__(self, tag, format = 'xml'):
        self.tag = tag
        self.format = format.lower()
        assert format in ['xml', 'html', 'json'], f"unknown format: {self.format}"

    def serialize(self, decoded_pred):
        if self.format == 'html':
            s = HTMLTagger(self.tag)
        elif self.format == 'xml': # elif self.format == 'xml':
            s = XMLTagger(self.tag)
        elif self.format == 'json':
            s = JSONTagger(self.tag)
        return s.serialize(decoded_pred)
