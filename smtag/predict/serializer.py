# -*- coding: utf-8 -*-
#T. Lemberger, 2018

#from abc import ABC
import torch
import xml.etree.ElementTree as ET
from ..common.utils import xml_escape, timer
from ..common.mapper import Catalogue, Boundary


class AbstractElementSerializer(object): # (ABC)

    def make_element(self, inner_text):
        pass

    def mark_boundary(self, action):
        pass

    def map(self, tag, on_features, concept):
        pass

class XMLElementSerializer(AbstractElementSerializer):

    @staticmethod
    def make_element(tag, on_features, inner_text):

        attribute_list= {}

        for concept in on_features: # what if a features correspond to mixed concepts? features should be linked to set of concepts or there should be a way to 'fuse' features.
            if concept:
                for attribute, value in XMLElementSerializer.map(concept):
                    # sometimes prediction is ambiguous and several values are found for a type or a role
                    if attribute in attribute_list and value != attribute_list[attribute]:
                        attribute_list[attribute] = "{}_{}".format(attribute_list[attribute], value) # not sure this is so great
                    else:
                        attribute_list[attribute] = value
        xml_attributes = ['{}="{}"'.format(a, attribute_list[a]) for a in attribute_list]
        xml_string = "<{} {}>{}</{}>".format(tag, ' '.join(xml_attributes), inner_text, tag)
        return xml_string # because both HTML and XML handled, working with strings is easier than return ET.tostring(xml_string)

    @staticmethod
    def mark_boundary(action): # in the future, it will accept a specific boundary but now only PANEL
        if action == 'open':
            return '<{}>'.format(XMLElementSerializer.map(Catalogue.PANEL_START)) # discrepancy: here via Catalogue, whereas via param tag for elements
        elif action == 'close':
            return '</{}>'.format(XMLElementSerializer.map(Catalogue.PANEL_START))

    @staticmethod
    def map(concept):
        #return entity_serializing_map[concept]
        return concept.for_serialization


class HTMLElementSerializer(AbstractElementSerializer):

    @staticmethod
    def make_element(tag, on_features, inner_text):
        html_classes = []
        html_class = ""
        for concept in on_features:
            if concept:
                for attribute, value in HTMLElementSerializer.map(concept):
                     html_class = "_".join([attribute, value])
                     if html_class not in html_classes:
                         html_classes.append(html_class)
        html_string = "<span class=\"{} {}\">{}</span>".format(tag, ' '.join(html_classes), inner_text)
        return html_string

    @staticmethod
    def mark_boundary(action): # in the future, it will accept a specific boundary but now only PANEL
        if action == 'open':
            return '<span class="{}">'.format(XMLElementSerializer.map(Catalogue.PANEL_START))
        elif action == 'close':
            return '</span>'


    @staticmethod
    def map(concept):
        return concept.for_serialization


class AbstractSerializer(object): #(ABC)

    def __init__(self, tag):
        self.tag = tag
        self.serialized_examples = []

    def serialize(self, binarized):
        self.N = binarized.N
        self.L = binarized.L
        self.nf = binarized.nf
        self.results = []
        self.output_semantics = binarized.output_semantics # an ordered sequence of Concept representing the concepts encoded by the output; needed to serialize it in XMl/HTML/Brag etc..

class AbstractTagger(AbstractSerializer):

    def __init__(self, tag):
        super(AbstractTagger, self).__init__(tag)

    def serialize_element(self, inner_text, on_features):
        pass

    def serialize_boundary(self, action):
        pass

    def serialize(self, binarized): # binarized contains N examples
        super(AbstractTagger, self).serialize(binarized)
        if Catalogue.PANEL_START in binarized.output_semantics:
            panel_feature = binarized.output_semantics.index(Catalogue.PANEL_START)
        else:
            panel_feature = None

        for i in range(self.N):
            #example_text = binarized.example_text[i]
            ml_string = ""
            inner_text = ""
            current_concepts = [False] * self.nf # usage: [map(concept) for concept in on_features if concept]
            need_to_open = [False] * self.nf
            need_to_close = [False] * self.nf
            need_to_open_any = False
            need_to_close_any = False
            active_features = 0
            current_scores = [0] * self.nf
            boundaries = torch.Tensor()
            if panel_feature is not None:
                # find where the panel boundaries are. Not very general but simpler than multiple hierarchical boundary types
                boundaries = binarized.start[ i , panel_feature , :].nonzero() # carefule: nonzero() return a list of coordinates of non zero element in the Tensor.

            token_start_positions = binarized.tokenized[i]['start_index']
            #print("token_start_positions", token_start_positions)
            #print("token list", " ".join([t.text for t in binarized.tokenized[i]['token_list']]))
            # segment example based on boundaries

            start = token_start_positions[0]
            segments = []
            if len(boundaries.nonzero()) != 0: # if example i has a position where boundaries[i] is 1, we need to segment
                for b in boundaries: # what if multiple kind of boundaries? this would be too complicated for the moment!!
                    if b in token_start_positions:
                        #find index of corresponding token
                        next_token_index = token_start_positions.index(b)
                        # add token from previous start to token before next
                        segment = binarized.tokenized[i]['token_list'][start:next_token_index]
                        segments.append(segment)
                        start = next_token_index

            last_segment = binarized.tokenized[i]['token_list'][start:]
            #print("last segment: ", " ".join([t.text for t in last_segment]))
            segments.append(last_segment)

            for token_list in segments:
            # change this to binarized.start.sum(1).nonzero etc then identify which feature is on; same for stop
            # change this to to start in benarized.tokenized.start_index to get immediately only the position where tag needs to be genearated: much faster!
                #print("serialize segment", " ".join([t.text for t in token_list]))
                #WARNING: CHECK BEFORE WHETHER ANY BOUNDARY FEATURES; IF NOT, DO NOT FLANK WITH <sd-panel>
                if panel_feature is not None:
                    ml_string += "<sd-panel>" # self.serialize_boundary('open') # problem: left spacer should be put before that
                for t in token_list:
                    start = t.start
                    stop = t.stop-1
                    left_spacer = t.left_spacer # left_spacer is the spacing characters that precede this token
                    text = xml_escape(t.text)

                    if active_features > 0:
                        inner_text +=  left_spacer
                    else:
                        ml_string += left_spacer

                    # scan features that need to be opened
                    for f in range(self.nf):
                        if not isinstance(self.output_semantics[f], Boundary) and binarized.start[i][f][start] != 0:
                            need_to_open[f] = True
                            need_to_open_any = True
                            active_features += 1

                    # as soon as something new needs to be opened all the rest needs to be closed first with the accumulated inner text
                    if need_to_open_any:
                        if inner_text:
                            tagged_string = self.serialize_element(current_concepts, inner_text)
                        else:
                            tagged_string =''
                        inner_text = ''
                        ml_string += tagged_string
                        for f in range(self.nf):
                            if  not isinstance(self.output_semantics[f], Boundary) and need_to_open[f]:
                                need_to_open[f] = False
                                concept = self.output_semantics[f]
                                current_concepts[f] = concept
                                current_scores[f] = binarized.score[i][f][start]
                        need_to_open_any = False

                    if active_features > 0:
                        inner_text +=  text
                    else:
                        ml_string += text

                    #scan features that need to be closed
                    for f in range(self.nf):
                        if  not isinstance(self.output_semantics[f], Boundary) and binarized.stop[i][f][stop] != 0:
                            need_to_close[f] = True
                            need_to_close_any = True
                            active_features -= 1

                    if need_to_close_any:
                        if inner_text:
                            tagged_string = self.serialize_element(current_concepts, inner_text)
                        else:
                            tagged_string = ''
                        inner_text = ''
                        ml_string += tagged_string
                        for f in range(self.nf):
                            if  not isinstance(self.output_semantics[f], Boundary) and need_to_close[f]:
                                need_to_close[f] = False
                                current_concepts[f] = False
                                current_scores[f] = 0
                        need_to_close_any = False
                if panel_feature is not None:
                    ml_string += "</sd-panel>" #self.serialize_boundary('close')
            #phew!
            self.serialized_examples.append(ml_string)
        return self.serialized_examples

class XMLTagger(AbstractTagger):

    def __init__(self, tag):
        super(XMLTagger, self).__init__(tag)

    def serialize_element(self, on_features, inner_text):
        return XMLElementSerializer.make_element(self.tag, on_features, inner_text)

    def serialize_boundary(self, action): # preliminary naive implementation...
        return XMLElementSerializer.mark_boundary(action)

    def serialize(self, binarized_pred):
        xml_string_list = super(XMLTagger, self).serialize(binarized_pred)
        # need to provide valid xml
        return ["<smtag>{}</smtag>".format(xml_string) for xml_string in xml_string_list]

class HTMLTagger(AbstractTagger):

    def __init__(self, tag):
        super(HTMLTagger, self).__init__(tag)

    def serialize_element(self, on_features, inner_text):
        return HTMLElementSerializer.make_element(self.tag, on_features, inner_text)

class BratSerializer(AbstractSerializer):

    def __init__(self, tag):
        super(BratSerializer, self).__init__(self, tag)

    def serialize_element(self, inner_text, on_features, scores): #need the positions; maybe does not need to be abstract method if I cannot change the parameters
        pass


class Serializer():

    def __init__(self, tag, format = 'xml'):
        self.tag = tag
        self.format = format.lower()

    def serialize(self, binarized_pred):
        if self.format == 'html':
            s = HTMLTagger(self.tag)
        elif self.format == 'xml':
            s = XMLTagger(self.tag)
        elif self.format == 'brat':
            s = BratSerializer(self.tag)
        return s.serialize(binarized_pred)
