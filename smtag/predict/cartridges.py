import torch
from torch import nn
from typing import List
from ..common.importexport import load_smtag_model
from ..common.mapper import Catalogue
from collections import OrderedDict
from .. import config


class CombinedModel(nn.Module):
    '''
    This module takes a list of SmtagModels and concatenates their output along the second dimension (feature axe).
    The output_semantics keeps track of the semantics of each module in the right order.
    '''

    def __init__(self, models: OrderedDict):
        super(CombinedModel, self).__init__()
        self.semantic_groups = OrderedDict()
        self.model_list = nn.ModuleDict(models)
        self.semantic_groups = {g:models[g].output_semantics for g in models}

    def forward(self, x):
        y_list = []
        for group in self.semantic_groups:
            model = self.model_list[group]
            y_list.append(model(x))
        y = torch.cat(y_list, 1)        
        return y

class ContextCombinedModel(nn.Module):

    def __init__(self, models: OrderedDict):
        super(ContextCombinedModel, self).__init__()
        self.semantic_groups = OrderedDict()
        self.anonymize_with = []
        self.model_list = nn.ModuleList()
        for group in models:
            model, anonymization = models[group]
            self.anonymize_with.append(anonymization)
            self.model_list.append(model)
            self.semantic_groups[group] = model.output_semantics
        #super(ContextCombinedModel, self).__init__(self.model_list) # PROBABLY WRONG: each model needs to be run on different anonymization input

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor: # takes a list of inputs each specifically anonymized for each context model
        y_list = [] 
        for m, x in zip(self.model_list, x_list):
            y_list.append(m(x))
        y = torch.cat(y_list, 1)
        return y


class Cartridge():

    def __init__(self, entity_models: CombinedModel, reporter_models: CombinedModel, context_models:ContextCombinedModel, panelize_model:CombinedModel):
        self.entity_models = entity_models
        self.reporter_models = reporter_models
        self.context_models = context_models
        self.panelize_model = panelize_model


models = {
    'reporter':             load_smtag_model(config.model_geneprod_reporter, config.prod_dir),
    'panel':                load_smtag_model(config.model_panel_stop,        config.prod_dir),
    'entity':               load_smtag_model(config.model_entity,            config.prod_dir),
    'geneprod_role':        load_smtag_model(config.model_geneprod_role,     config.prod_dir),
    'molecule_role':        load_smtag_model(config.model_molecule_role,     config.prod_dir),
    'disease':              load_smtag_model(config.model_disease,           config.prod_dir),
}

# put models on GPU DataParallel when possible
if torch.cuda.is_available():
    print(f"{torch.cuda.device_count()} GPUs available for model cartridge.")
    for m in models:
        models[m] = models[m].cuda()
        gpu_model = nn.DataParallel(models[m])
        gpu_model.output_semantics = models[m].output_semantics
        models[m] = gpu_model

CARTRIDGE = Cartridge(
    entity_models = CombinedModel(OrderedDict([
        ('entities', models['entity']),
        ('diseases', models['disease']),
    ])),
    reporter_models = CombinedModel(OrderedDict([
        ('reporter', models['reporter']),
    ])),
    context_models = ContextCombinedModel(OrderedDict([
        ('geneprod_roles',
             (models['geneprod_role'], {'group': 'entities', 'concept': Catalogue.GENEPROD}),
        ),
        ('small_molecule_role',
            (models['molecule_role'], {'group': 'entities', 'concept': Catalogue.SMALL_MOLECULE}),
        ),
    ])),
    panelize_model = CombinedModel(OrderedDict([
        ('panels', models['panel']),
    ])),
)