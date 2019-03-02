# -*- coding: utf-8 -*-
#T. Lemberger, 2018

# SmartTag learns to recognize a number of features in stretches of text. For example if a strech of text corresponds to a gene product,
# a disease or if a new figure panel legends is starting, the respective features will be recognized by SmartTag.
# For training, features are typically learned from compendia where the features were tagged, for example using xml tags.
# At prediction stage, the recognized features can be serialized back into xml (or html).
# We have therefore the back and forther conversion from XML to tensor to XML:
# XML --> train features(text) // predict features(text) --> XML rendering.
#
# A feature can either be a set of longitudinal marks under each character of the word/text that possess the feature of interest. 
# This is the system used for entitiy recognition.
# Alternatively, it can be a boundary that defines the begining or the end of a segment of text. 
# This is the system used to recognize the start of a figure panel.
# To transform xml into stacked features (or channels), specific combinations of element/attribute/value are mapped 
# onto the index of the feature that represents this particular feature (1 or 0 in a torch.Tensor).
# The tensor that holds the features is 3D: N examples X nf features (or channels) x L characters (string length).
# For example the feature "organism" will mark the word "cat" and the feature "tissue" will mark "heart" in the sentence below:
#
#                  The cat with a heart.
# SMALL_MOLECULE   000000000000000000000
#    ...
# TISSUE           000000000000000111110
# ORGANISM         000011100000000000000
#   ...
#
# This tensor is is extracted from the xml:
#     <xml><p>The <sd-tag type='organism'>cat</sd-tag> with a <sd-tag type='tissue'>heart</sd-tag>.</xml>
# by mapping the element sd-tag with attribute @type='organism' to the feature (channel) with index 6 (zero based).
# The mapping is given in xml_map. The xml to feature conversion is carried out in the module smtag.featurizer
#
# The reverse operation is made possible by the class Concept and Catalogue.
# Catalogue provides the list of the concepts learned by SmartTag in the same order as in xml_map. Concepts can be an Entity, a generic Element or a Boundary.
# To each concept, the information necessary for xml or html serialization is provided as a 'recipe' in the form of
# attribute-value pair for Elements and Entity or as the name of the element that will represent a Boundary.
# Each features has a threshold of detection. This can be treated as hyperparameters that can be tuned and optimized after training.
# This is still messy and problematic. Ultimately Concept, Catalogue and xml_map have to be unified/merged into a single object and json description.
# Element, Entity and Boundary are used in smtag.serializer


class Concept(object):
    def __init__(self, label = "", recipe = [], detection_threshold = 0.5):
        self.label = label
        self.for_serialization = recipe
        self.threshold = detection_threshold

    def __str__(self):
        return f"{self.label}"

    def __repr__(self):
        return f"<{self.label}>"

    def my_index(self, list):
        for i, c in enumerate(list):
            if type(self) == type(c):
                return i
        return None

class Untagged(Concept):
    def __init__(self, *args):
        super(Untagged, self).__init__(*args)

class Category(Concept):
    def __init__(self, *args):
        super(Category, self).__init__(*args)

class Disease(Category):
    def __init__(self, *args):
        super(Disease, self).__init__(*args)

class TimeVar(Category):
    def __init__(self, *args):
        super(TimeVar, self).__init__(*args)

class PhysicalVar(Category):
    def __init__(self, *args):
        super(PhysicalVar, self).__init__(*args)

class Assay(Category):
    def __init__(self, *args):
        super(Assay, self).__init__(*args)

class Entity(Category):
    def __init__(self, *args):
        super(Entity, self).__init__(*args)

class SmallMolecule(Entity):
    def __init__(self, *args):
        super(SmallMolecule, self).__init__(*args)

class Gene(Entity):
    def __init__(self, *args):
        super(Gene, self).__init__(*args)

class Protein(Entity):
    def __init__(self, *args):
        super(Protein, self).__init__(*args)

class GeneProduct(Entity):
    def __init__(self, *args):
        super(GeneProduct, self).__init__(*args)

class Subcellular(Entity):
    def __init__(self, *args):
        super(Subcellular, self).__init__(*args)

class Cell(Entity):
    def __init__(self, *args):
        super(Cell, self).__init__(*args)

class Tissue(Entity):
    def __init__(self, *args):
        super(Tissue, self).__init__(*args)

class Organism(Entity):
    def __init__(self, *args):
        super(Organism, self).__init__(*args)

class Undefined(Entity):
    def __init__(self, *args):
        super(Undefined, self).__init__(*args)

class Role(Category):
    def __init__(self, *args):
        super(Role, self).__init__(*args)

class Intervention(Role):
    def __init__(self, *args):
        super(Intervention, self).__init__(*args)

class Measurement(Role):
    def __init__(self, *args):
        super(Measurement, self).__init__(*args)

class Reporter(Role):
    def __init__(self, *args):
        super(Reporter, self).__init__(*args)

class ExperimentalVar(Role):
    def __init__(self, *args):
        super(ExperimentalVar, self).__init__(*args)

class Normalizing(Role):
    def __init__(self, *args):
        super(Normalizing, self).__init__(*args)

class Generic(Role):
    def __init__(self, *args):
        super(Generic, self).__init__(*args)

class Boundary(Concept):
    def __init__(self, *args):
        super(Boundary, self).__init__(*args)

class PanelStart(Boundary):
    def __init__(self, *args):
        super(PanelStart, self).__init__(*args)

class PanelStop(Concept):
    def __init__(self, *args):
        super(PanelStop, self).__init__(*args)

class Catalogue():

    SMALL_MOLECULE = SmallMolecule('small_molecule', ['type', 'small_molecule'], 0.6)
    GENE = Gene('gene', ['type', 'gene'], 0.4)
    PROTEIN = Protein('protein', ['type', 'protein'], 0.4)
    SUBCELLULAR = Subcellular('subcellular', ['type', 'subcellular'], 0.4)
    CELL = Cell('cell', ['type', 'cell'], 0.5)
    TISSUE = Tissue('tissue', ['type', 'tissue'], 0.4)
    ORGANISM = Organism('organism', ['type', 'organism'], 0.6)
    UNDEFINED = Undefined('undefined', ['type', 'undefined'], 0.5)
    INTERVENTION = Intervention('intervention', ['role', 'intervention'], 0.4)
    MEASUREMENT = Measurement('assayed', ['role', 'assayed'], 0.4)
    NORMALIZING = Normalizing('normalizing', ['role', 'normalizing'], 0.5)
    REPORTER = Reporter('reporter', ['role', 'reporter'], 0.8)
    EXP_VAR = ExperimentalVar('experiment', ['role', 'experiment'], 0.5)
    GENERIC = Generic('component', ['role', 'component'], 0.5)
    EXP_ASSAY = Assay('assay', ['category', 'assay'], 0.6)
    ENTITY = Entity('entity', ['category', 'entity'], 0.5)
    TIME = TimeVar('time', ['category', 'assay'], 0.5)
    PHYSICAL_VAR = PhysicalVar('physical', ['category', 'physical'], 0.5)
    DISEASE = Disease('disease', ['category', 'disease'], 0.5)
    PANEL_START = PanelStart('panel_start','sd-panel',  0.5)
    PANEL_STOP = PanelStop('panel_stop', 'sd-panel', 0.5) # not ideal!
    GENEPROD = GeneProduct('geneprod', ['type', 'geneprod'], 0.4)
    UNTAGGED = Untagged('untagged', [])

    # the order of the Concepts in the catalogue matters and determines the order in which these concepts are expected in datasets used for training
    standard_channels = [SMALL_MOLECULE, GENE, PROTEIN, SUBCELLULAR, CELL, TISSUE, ORGANISM, UNDEFINED,
            INTERVENTION, MEASUREMENT, NORMALIZING, REPORTER, EXP_VAR, GENERIC,
            EXP_ASSAY, ENTITY, TIME, PHYSICAL_VAR, DISEASE, PANEL_START, PANEL_STOP, GENEPROD]


    @staticmethod
    def from_list(labels):
        return [Catalogue.from_label(label) for label in labels]

    @staticmethod
    def from_label(label):
        if label:
            return label2concept[label] # OK to raises error if label not in label2concept?
        else:
            return None

NUMBER_OF_ENCODED_FEATURES = len(Catalogue.standard_channels)

label2concept = {e.label:e for e in Catalogue.standard_channels}
concept2index = {Catalogue.standard_channels[i]:i for i in range(len(Catalogue.standard_channels))}
index2concept = {i:Catalogue.standard_channels[i] for i in range(len(Catalogue.standard_channels))}


brat_map = {'': None, 'DISO': 18, 'PRGE': 21, 'GENE': 1, 'LIVB': 6, 'CHED':0} # check if PRGE should go to channel 21

# xml_map should be the 'master' description of the model
# Eventually, the Concept class and subclasses and the catalogue should be generated automatically from this description
# xml_map should be read from semantic_model.json
xml_map = {
            'marks':{
                    'sd-tag':{
                            'type':{'':None, 'molecule':0, 'gene':1, 'protein':2, 'geneprod':21, 'subcellular':3, 'cell':4, 'tissue':5, 'organism':6, 'undefined':7},
                            'role':{'':None, 'intervention':8, 'assayed':9, 'normalizing':10, 'reporter':11, 'experiment':12, 'component':13},
                            'category':{'':None, 'assay':14, 'entity':15, 'time':16, 'physical':17, 'disease':18}
                            }
                    },

            'boundaries':{
                        'sd-panel':{
                                    '': {'':[19,20]}
                                    }
                         }
           }
