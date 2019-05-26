from ..datagen.context import VisualContext
from .engine import Cartridge, CombinedModel, ContextCombinedModel
from ..common.importexport import load_model
from ..common.mapper import Catalogue
from collections import OrderedDict
from .. import config


entity_viz = load_model(config.model_entity_viz, config.prod_dir)
disease = load_model(config.model_disease_no_viz, config.prod_dir)
reporter = load_model(config.model_geneprod_reporter_no_viz, config.prod_dir)
geneprod_role_viz = load_model(config.model_geneprod_role_viz, config.prod_dir)
molecule_role_viz = load_model(config.model_molecule_role_viz, config.prod_dir)
panel = load_model(config.model_panel_stop_no_viz, config.prod_dir)
entity_no_viz = load_model(config.model_entity_no_viz, config.prod_dir)
geneprod_role_no_viz = load_model(config.model_geneprod_role_no_viz, config.prod_dir)
molecule_role_no_viz = load_model(config.model_molecule_role_no_viz, config.prod_dir)

WITH_VIZ = Cartridge(
    entity_models = CombinedModel(OrderedDict([
        ('entities', entity_viz),
        #('diseases', disease),
    ])),
    reporter_models = CombinedModel(OrderedDict([
        ('reporter', reporter),
    ])),
    context_models = ContextCombinedModel(OrderedDict([
        ('geneprod_roles',
            (geneprod_role_viz, {'group': 'entities', 'concept': Catalogue.GENEPROD}),
        ),
        ('small_molecule_role',
            (molecule_role_viz, {'group': 'entities', 'concept': Catalogue.SMALL_MOLECULE}),
        )
    ])),
    panelize_model = CombinedModel(OrderedDict([
        ('panels', panel),
    ])),
    viz_preprocessor = VisualContext(),
)


NO_VIZ = Cartridge(
    entity_models = CombinedModel(OrderedDict([
        ('entities', entity_no_viz),
        #('diseases', disease),
    ])),
    reporter_models = CombinedModel(OrderedDict([
        ('reporter', reporter),
    ])),
    context_models = ContextCombinedModel(OrderedDict([
        ('geneprod_roles',
            (geneprod_role_no_viz, {'group': 'entities', 'concept': Catalogue.GENEPROD}),
        ),
        ('small_molecule_role',
            (molecule_role_no_viz, {'group': 'entities', 'concept': Catalogue.SMALL_MOLECULE}),
        )
    ])),
    panelize_model = CombinedModel(OrderedDict([
        ('panels', panel),
    ])),
    viz_preprocessor = VisualContext(),
)