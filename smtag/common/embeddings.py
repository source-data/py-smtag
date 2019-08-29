from .importexport import load_model
from .catstack import CatStack
from .. import config
if config.embeddings_model:
    EMBEDDINGS = load_model(config.embeddings_model, config.embeddings_dir, CatStack)
else:
    EMBEDDINGS = None