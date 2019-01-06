
from xml.etree.ElementTree import fromstring

DEFAULT_PRETAG = fromstring('<sd-tag/>')

def updatexml_(xml, bin_pred, position=0, pretag=DEFAULT_PRETAG):

    # only update pretagged elements as specificed by pretag
    if xml.tag == pretag.tag: 
        required_attributes = True
        # this allows to use pretag to update only element that have the same tag and some required attributes set to specific values
        # example: "<sd-tag type='geneprod'/>" to only update geneproduct tags
        for a in pretag.attrib: 
            required_attributes = xml.attrib[a] == pretag.attrib[a] and required_attributes 

        if required_attributes:
            max_score = {}
            for i, concept in enumerate(bin_pred.output_semantics):
                # DANGER: context-dep semantics depends on entity type; what if role for wrong type or wrong category for type, role?
                for attribute, value in concept.for_serialization: # attribute, value, default
                    if not attribute in max_score:
                        max_score[attribute] = 0
                    if bin_pred.score[0, i, position] > concept.threshold and bin_pred.score[0, i, position] > max_score[attribute]:  
                        # overwrite if predicted score above threshold and above previous best score for same attribute
                        xml.attrib[attribute] = value
                        max_score[attribute] = bin_pred.score[0, i, position]
                        score_attribute = attribute + "_score"
                        score_value = str(int(bin_pred.score[0, i, position].item()))
                        xml.attrib[score_attribute] = score_value
                    # elif not attribute in xml.attrib: 
                    #    xml.attrib[attribute] = default

    if xml.text is not None:
        position += len(xml.text)    
    for child in xml:
        # RECURSIVE CALL ON EACH CHILDREN
        position = updatexml_(child, bin_pred, position, pretag)
    if xml.tail is not None:
        position += len(xml.tail)
    return position