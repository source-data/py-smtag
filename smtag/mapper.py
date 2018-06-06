#maybe should be called channel_to_concept and concept_to_channel
#should these rather be named tuples of named tuples?

index2label = {0: 'small_molecule', 1: 'gene', 2: 'protein', 3: 'subcellular', 4: 'cell', 5: 'tissue', 6: 'organism', 7: 'undefined',
               8: 'intervention', 9: 'assayed', 10: 'normalizing', 11: 'reporter', 12: 'experiment', 13: 'component',
               14: 'assay', 15: 'entity', 16: 'time', 17: 'physical', 18: 'disease',
               19: 'panel_start', 20: 'panel_stop', 21: 'geneprod'}
               
label2index = {index2label[k]:k for k in index2label}

brat_map = {'': None, 'DISO': 18, 'PRGE': 21, 'GENE': 1, 'LIVB': 6, 'CHED':0} # check if PRGE should go to channel 21

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

number_of_features = len(index2label)

THRESHOLDS = {
        'geneprod': 0.7
}