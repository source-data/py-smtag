# -*- coding: utf-8 -*-
#T. Lemberger, 2018

from ..common.mapper import brat_map, xml_map

#features should be instance of Features()
#define abstract Featurizer class that takes Example and returns Features

class AnnFeaturizer(object):

    @staticmethod
    def ann2features(example):
        L = len(example['text'])
        annot = example['annot']
        features = {'marks':{'ann':{'type':[None] * L}}}
        for a in annot:
            start = a['start']
            stop = a['stop']
            type = a['type']
            code = brat_map[type]
            for i in range(start, stop):
                features['marks']['ann']['type'][i] = code
        return features

class XMLEncoder(object):

    @staticmethod
    def featurize_marks(element, L, features = {}):
        element_tag = element.tag
        #initialization if no features were coded before from a parent element
        if not features:
            features = {kind:{el:{attr:[None] * L for attr in xml_map[kind][el]} for el in xml_map[kind]} for kind in xml_map}

        if element_tag in xml_map['marks']:
            if '' in xml_map['marks'][element_tag]:
                features['marks'][element_tag][''] = [xml_map['marks'][element_tag]['']['']] * L

            for attribute in (set(xml_map['marks'][element_tag].keys()) & set(element.attrib)):
                val = element.attrib[attribute]
                if val and val in xml_map['marks'][element_tag][attribute]:
                    features['marks'][element_tag][attribute] = [xml_map['marks'][element_tag][attribute][val]] * L
        return features

    @staticmethod
    def featurize_boundaries(element, features, L):
        element_tag = element.tag
        #L = len(features['boundaries']['sd-panel'][''])
        #features['boundaries'] = {el:{attr:[None] * L for attr in xml_map['boundaries'][el]} for el in xml_map['boundaries']}

        if element_tag in xml_map['boundaries'] and L > 0:
            if '' in xml_map['boundaries'][element_tag]:
                features['boundaries'][element_tag][''][0] = xml_map['boundaries'][element_tag][''][''][0]
                features['boundaries'][element_tag][''][L-1] = xml_map['boundaries'][element_tag][''][''][1]


            for attribute in (set(xml_map['boundaries'][element_tag].keys()) & set(element.attrib)):
                val = element.attrib[attribute]
                if val and val in xml_map['boundaries'][element_tag][attribute]:
                    features['boundaries'][element_tag][attribute][0] = xml_map['boundaries'][element_tag][attribute][val][0]
                    features['boundaries'][element_tag][attribute][L-1] = xml_map['boundaries'][element_tag][attribute][val][1]
        return features

    @staticmethod
    def encode(element):
        features = {kind:{el:{attr:[] for attr in xml_map[kind][el]} for el in xml_map[kind]} for kind in xml_map}

        if element is not None:
            text_core = element.text or ''
            L_core = len(text_core)
            text_tail = element.tail or ''
            L_tail = len(text_tail)
            features = XMLEncoder.featurize_marks(element, L_core)
            L_tot = L_core

            #add marks recursively
            for child in list(element):
                child_core, L_child_core, L_child_tail = XMLEncoder.encode(child)
                L_tot = L_tot + L_child_core + L_child_tail
                #adding to child the features inherited from parent element
                child_core = XMLEncoder.featurize_marks(element, L_child_core, child_core)
                child_tail = XMLEncoder.featurize_marks(element, L_child_tail)

                for kind in features:
                    for e in features[kind]:
                            for a in features[kind][e]:
                                features[kind][e][a] += child_core[kind][e][a] + child_tail[kind][e][a]

            #add boundaries to current element
            try:
                features = XMLEncoder.featurize_boundaries(element, features, L_tot)
            except Exception as e:
                print(element.text, element.tag, element.attrib)
                print(features)
                print(L_tot)
                raise(e)

            #add 'virtual' computed features here?

        return features, L_tot, L_tail
