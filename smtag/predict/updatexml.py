
from xml.etree.ElementTree import fromstring
from xml.etree.ElementTree import Element
from typing import List
from .decode import Decoder
from ..common.mapper import Catalogue, Concept

DEFAULT_PRETAG = fromstring('<sd-tag/>')

# def updatexml_(xml: Element, semantic_groups: str, char_level_concepts: List[Concept], position=0, pretag=DEFAULT_PRETAG):

#     # only update pretagged elements as specificed by pretag
#     if xml.tag == pretag.tag: 
#         required_attributes = True
#         # this allows to use pretag to update only element that have the same tag and some required attributes set to specific values
#         # example: "<sd-tag type='geneprod'/>" to only update geneproduct tags
#         for a in pretag.attrib: 
#             required_attributes = xml.attrib[a] == pretag.attrib[a] and required_attributes 
#         if required_attributes:
#             for group in semantic_groups:
#                 concept = char_level_concepts[group][position]
#                 if concept is not None and concept != Catalogue.UNTAGGED:
#                     attribute, value = concept.for_serialization
#                     xml.attrib[attribute] = value
#     if xml.text is not None:
#         position += len(xml.text)    
#     for child in xml:
#         # RECURSIVE CALL ON EACH CHILDREN
#         position = updatexml_(child, semantic_groups, char_level_concepts, position, pretag)
#     if xml.tail is not None:
#         position += len(xml.tail)
#     return position

def updatexml_list(xml_list: List[Element], decoded: Decoder, pretag=DEFAULT_PRETAG):
    assert len(xml_list)==decoded.N
    for n in range(decoded.N):
        def updatexml_(xml, position=0):
            # only update pretagged elements as specificed by pretag
            if xml.tag == pretag.tag: 
                required_attributes = True
                # this allows to use pretag to update only element that have the same tag and some required attributes set to specific values
                # example: "<sd-tag type='geneprod'/>" to only update geneproduct tags
                for a in pretag.attrib: 
                    required_attributes = xml.attrib[a] == pretag.attrib[a] and required_attributes 
                if required_attributes:
                    for group in decoded.semantic_groups:
                        concept = decoded.char_level_concepts[n][group][position]
                        if concept is not None and concept != Catalogue.UNTAGGED:
                            attribute, value = concept.for_serialization
                            xml.attrib[attribute] = value
            if xml.text is not None:
                position += len(xml.text)    
            for child in xml:
                # RECURSIVE CALL ON EACH CHILDREN
                position = updatexml_(child, position)
            if xml.tail is not None:
                position += len(xml.tail)
            return position
        updatexml_(xml_list[n])
    return xml_list
