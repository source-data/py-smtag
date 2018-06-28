# -*- coding: utf-8 -*-
#T. Lemberger, 2018

TYPES = ['molecule', 'gene', 'protein', 'geneprod', 'subcellular', 'cell', 'tissue', 'organism', 'undefined']
ROLES = ['intervention', 'assayed', 'reporter']
CATEGORIES = ['entities', 'exp_assay', 'physical', 'time', 'disease']
BOUNDARIES = ['panel_start', 'panel_stop']

class Concept:
    def __init__(self, label, recipe, detection_threshold = 0.5):
        self.label = label
        self.for_serialization = recipe
        self.threshold = detection_threshold
    
    def __str__(self):
        return self.label

class Element(Concept):
    def __init__(self, label, recipe, detection_threshold = 0.5):
        super(Element, self).__init__(detection_threshold, recipe)

class Entity(Element):

    def __init__(self, label, recipe, detection_threshold = 0.5):
        super(Entity, self).__init__(detection_threshold, recipe)

class Boundary(Concept):
    def __init__(self, label, recipe, detection_threshold = 0.5):
        super(Boundary, self).__init__(detection_threshold, recipe)

SMALL_MOLECULE = Entity('small_molecule', 0.5, ('type', 'small_molecule'))
GENE = Entity('gene', 0.5, ('type', 'gene'))
PROTEIN = Entity('protein', 0.5, ('type', 'protein'))
SUBCELLULAR = Entity('subcellular', 0.5, ('type', 'subcellular'))
CELL = Entity('cell', 0.5, ('type', 'cell'))
TISSUE = Entity('tissue', 0.5 ('type', 'tissue'))
ORGANISM = Entity('organism', 0.5, ('type', 'organism'))
UNDEFINED = Entity('undefined', 0.5, ('type', 'undefined'))
INTERVENTION = Entity('intervention', 0.5, ('role', 'intervention'))
MEASUREMENT = Entity('assayed', 0.5, ('role', 'assayed'))
NORMALIZING = Entity('normalizing', 0.5, ('role', 'normalizing'))
REPORTER = Entity('reporter', 0.5, ('role', 'reporter'))
EXP_VAR = Entity('experiment', 0.5, ('role', 'experiment'))
GENERIC_ENTITY = Entity('component', 0.5, ('role', 'component'))
EXP_ASSAY = Element('assay', 0.5, ('category', 'assay'))
TIME = Element('time', 0.5, ('category', 'assay'))
PHYSICAL_VAR = Element('physical', 0.5, ('category', 'physical'))
DISEASE = Element('disease', 0.5, ('category', 'disease'))
PANEL_START = Boundary('panel_start', 0.5, 'sd-panel')
PANEL_STOP = Boundary('panel_stop', 0.5, 'sd-panel') # not ideal!
GENEPROD = Entity('geneprod', 0.5, ('type', 'geneprod'))

catalogue = [SMALL_MOLECULE, GENE, PROTEIN, SUBCELLULAR, CELL, TISSUE, ORGANISM, UNDEFINED,
             INTERVENTION, MEASUREMENT, NORMALIZING, REPORTER, EXP_VAR, GENERIC_ENTITY,
             EXP_ASSAY, TIME, PHYSICAL_VAR, DISEASE, PANEL_START, PANEL_STOP, GENEPROD]

label2concept = {e.label:e for e in catalogue}
concept2index = {catalogue[i]:i for i in range(len(catalogue))}
index2concept = {i:catalogue[i] for i in range(len(catalogue))}

class Factory():

    @staticmethod
    def from_list(labels):
        return [Factory.make(label) for label in labels]
    
    @staticmethod
    def make(label):
            return label2concept(label)

brat_map = {'': None, 'DISO': 18, 'PRGE': 21, 'GENE': 1, 'LIVB': 6, 'CHED':0} # check if PRGE should go to channel 21

# this should be the 'master' description of the model and the rest should be generated automatically from this description
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
