# -*- coding: utf-8 -*-
#T. Lemberger, 2018

class Concept(object):
    def __init__(self, label, recipe, detection_threshold = 0.5):
        self.label = label
        self.for_serialization = recipe
        self.threshold = detection_threshold
    
    def __str__(self):
        return self.label

class Element(Concept):
    def __init__(self, *args):
        super(Element, self).__init__(*args)

class Entity(Element):

    def __init__(self, *args):
        super(Entity, self).__init__(*args)

class Boundary(Concept):
    def __init__(self, *args):
        super(Boundary, self).__init__(*args)

class Catalogue():

    SMALL_MOLECULE = Entity('small_molecule', ('type', 'small_molecule'), 0.5)
    GENE = Entity('gene', ('type', 'gene'), 0.5)
    PROTEIN = Entity('protein', ('type', 'protein'), 0.5)
    SUBCELLULAR = Entity('subcellular', ('type', 'subcellular'), 0.5)
    CELL = Entity('cell', ('type', 'cell'), 0.5)
    TISSUE = Entity('tissue', ('type', 'tissue'), 0.5)
    ORGANISM = Entity('organism', ('type', 'organism'), 0.5)
    UNDEFINED = Entity('undefined', ('type', 'undefined'), 0.5)
    INTERVENTION = Entity('intervention', ('role', 'intervention'), 0.5)
    MEASUREMENT = Entity('assayed', ('role', 'assayed'), 0.5)
    NORMALIZING = Entity('normalizing', ('role', 'normalizing'), 0.5)
    REPORTER = Entity('reporter', ('role', 'reporter'), 0.5)
    EXP_VAR = Entity('experiment', ('role', 'experiment'), 0.5)
    GENERIC_ENTITY = Entity('component', ('role', 'component'), 0.5)
    EXP_ASSAY = Element('assay', ('category', 'assay'), 0.5)
    ENTITY = Element('entity', ('category', 'entity'), 0.5)
    TIME = Element('time', ('category', 'assay'), 0.5)
    PHYSICAL_VAR = Element('physical', ('category', 'physical'), 0.5)
    DISEASE = Element('disease', ('category', 'disease'), 0.5)
    PANEL_START = Boundary('panel_start','sd-panel',  0.5)
    PANEL_STOP = Boundary('panel_stop', 'sd-panel', 0.5) # not ideal!
    GENEPROD = Entity('geneprod', ('type', 'geneprod'), 0.5)
    
    # the order of the Concepts in the cataglogue matters and determine the order in which these concepts are expected in datasets used for training
    standard_channels = [SMALL_MOLECULE, GENE, PROTEIN, SUBCELLULAR, CELL, TISSUE, ORGANISM, UNDEFINED,
            INTERVENTION, MEASUREMENT, NORMALIZING, REPORTER, EXP_VAR, GENERIC_ENTITY,
            EXP_ASSAY, ENTITY, TIME, PHYSICAL_VAR, DISEASE, PANEL_START, PANEL_STOP, GENEPROD]


    @staticmethod
    def from_list(labels):
        return [Catalogue.from_label(label) for label in labels]
    
    @staticmethod
    def from_label(label):
            return label2concept[label]


label2concept = {e.label:e for e in Catalogue.standard_channels}
concept2index = {Catalogue.standard_channels[i]:i for i in range(len(Catalogue.standard_channels))}
index2concept = {i:Catalogue.standard_channels[i] for i in range(len(Catalogue.standard_channels))}


brat_map = {'': None, 'DISO': 18, 'PRGE': 21, 'GENE': 1, 'LIVB': 6, 'CHED':0} # check if PRGE should go to channel 21

# xml_map should be the 'master' description of the model
# Eventually, the Concept class and subclasses and the catalogue should be generated automatically from this description
# xml_map should be read from model.json
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
