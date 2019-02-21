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
# Another source of difficulty is that some features are combinations of several features (with And or Or).
# __add__() implements a way to fuse/merge concepts.
# There should be a way to search for a feature based on one attribute of the concept


class Concept(object):
    def __init__(self, label = "", recipe = [], detection_threshold = 0.5):
        self.label = label
        self.for_serialization = recipe
        self.threshold = detection_threshold
        self.category = ""
        self.type = ""
        self.role = ""

    def __str__(self):
        return "{}: '{}' ({})".format(self.category, self.label, "; ".join(filter(None, [self.type, self.role])))

    def __repr__(self):
        return "{}".format(self.label)

    def __add__(self, x): # maybe misleading because not commutative
        assert(self.category == x.category or self.category == "" or x.category == "") # cannot combine concepts across categories but can do with category-less object
        y = Concept()
        y.label = "_".join(filter(None, [self.label, x.label]))
        if self.category == "":
            y.category = x.category
        else:
            y.category = self.category
        if self.role != x.role:
            y.role = "_".join(filter(None, [self.role, x.role]))
        if self.type != x.type:
            y.type = "_".join(filter(None, [self.type, x.type]))
        y.threshold = (self.threshold + x.threshold) / 2
        y.for_serialization += x.for_serialization
        return y

    def equal_class(self, x):
        # we neglect differences in role to call it 'equal'; ok to find concept in list of entities but not very general.
        # return self.category == x.category and self.type == x.type
        return type(x) == type(self)

    def my_index(self, list):
        i = 0
        for c in list:
            if self.equal_class(c):
                return i
            i += 1
        if i == len(list):
            return None


class Category(Concept):
    def __init__(self, label, *args):
        super(Category, self).__init__(label, *args)
        self.category = label


class Entity(Category):
    def __init__(self, label, *args):
        super(Entity, self).__init__(label, *args)
        self.category = "entity"
        self.type = label


class Role(Category):
    def __init__(self, label, *args):
        super(Role, self).__init__(label, *args)
        self.category = "entity"
        self.role = label

class Boundary(Concept):
    def __init__(self, *args):
        super(Boundary, self).__init__(*args)

class Catalogue():

    SMALL_MOLECULE = Entity('small_molecule', ['type', 'small_molecule'], 0.6)
    GENE = Entity('gene', ['type', 'gene'], 0.4)
    PROTEIN = Entity('protein', ['type', 'protein'], 0.4)
    SUBCELLULAR = Entity('subcellular', ['type', 'subcellular'], 0.4)
    CELL = Entity('cell', ['type', 'cell'], 0.5)
    TISSUE = Entity('tissue', ['type', 'tissue'], 0.4)
    ORGANISM = Entity('organism', ['type', 'organism'], 0.6)
    UNDEFINED = Entity('undefined', ['type', 'undefined'], 0.5)
    INTERVENTION = Role('intervention', ['role', 'intervention'], 0.4)
    MEASUREMENT = Role('assayed', ['role', 'assayed'], 0.4)
    NORMALIZING = Role('normalizing', ['role', 'normalizing'], 0.5)
    REPORTER = Role('reporter', ['role', 'reporter'], 0.8)
    EXP_VAR = Role('experiment', ['role', 'experiment'], 0.5)
    GENERIC = Role('component', ['role', 'component'], 0.5)
    EXP_ASSAY = Category('assay', ['category', 'assay'], 0.6)
    ENTITY = Category('entity', ['category', 'entity'], 0.5)
    TIME = Category('time', ['category', 'assay'], 0.5)
    PHYSICAL_VAR = Category('physical', ['category', 'physical'], 0.5)
    DISEASE = Category('disease', ['category', 'disease'], 0.5)
    PANEL_START = Boundary('panel_start','sd-panel',  0.5)
    PANEL_STOP = Boundary('panel_stop', 'sd-panel', 0.5) # not ideal!
    GENEPROD = Entity('geneprod', ['type', 'geneprod'], 0.4)
    UNTAGGED = Concept('untagged', [])

    # the order of the Concepts in the catalogue matters and determines the order in which these concepts are expected in datasets used for training
    standard_channels = [SMALL_MOLECULE, GENE, PROTEIN, SUBCELLULAR, CELL, TISSUE, ORGANISM, UNDEFINED,
            INTERVENTION, MEASUREMENT, NORMALIZING, REPORTER, EXP_VAR, GENERIC,
            EXP_ASSAY, ENTITY, TIME, PHYSICAL_VAR, DISEASE, PANEL_START, PANEL_STOP, GENEPROD, UNTAGGED]


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
