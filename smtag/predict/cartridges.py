from ..datagen.context import VisualContext
from .engine import Cartridge, CombinedModel, ContextCombinedModel
from ..common.importexport import load_model
from collections import OrderedDict
from .. import config

CARTRIDGE_WITH_VIZ = Cartridge(
    entity_models = CombinedModel(OrderedDict([
        ('entities', load_model(config.model_entity_viz, config.prod_dir)),
        #('diseases', load_model(config.model_disease_no_viz, config.prod_dir)),
    ])),
    reporter_models = CombinedModel(OrderedDict([
        ('reporter', load_model(config.model_geneprod_reporter_no_viz, config.prod_dir)),
    ])),
    context_models = ContextCombinedModel(OrderedDict([
        ('geneprod_roles',
            (load_model(config.model_geneprod_role_viz, config.prod_dir), {'group': 'entities', 'concept': Catalogue.GENEPROD}),
        ),
        ('small_molecule_role',
            (load_model(config.model_molecule_role_viz, config.prod_dir), {'group': 'entities', 'concept': Catalogue.SMALL_MOLECULE}),
        )
    ])),
    panelize_model = CombinedModel(OrderedDict([
        ('panels', load_model(config.model_panel_stop_no_viz, config.prod_dir)),
    ])),
    viz_preprocessor = VisualContext(),
)


CARTRIDGE_NO_VIZ = Cartridge(
    entity_models = CombinedModel(OrderedDict([
        ('entities', load_model(config.model_entity_no_viz, config.prod_dir)),
        #('diseases', load_model(config.model_disease_no_viz, config.prod_dir)),
    ])),
    reporter_models = CombinedModel(OrderedDict([
        ('reporter', load_model(config.model_geneprod_reporter_no_viz, config.prod_dir)),
    ])),
    context_models = ContextCombinedModel(OrderedDict([
        ('geneprod_roles',
            (load_model(config.model_geneprod_role_no_viz, config.prod_dir), {'group': 'entities', 'concept': Catalogue.GENEPROD}),
        ),
        ('small_molecule_role',
            (load_model(config.model_molecule_role_no_viz, config.prod_dir), {'group': 'entities', 'concept': Catalogue.SMALL_MOLECULE}),
        )
    ])),
    panelize_model = CombinedModel(OrderedDict([
        ('panels', load_model(config.model_panel_stop_no_viz, config.prod_dir)),
    ])),
    viz_preprocessor = VisualContext(),
)