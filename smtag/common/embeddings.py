from .importexport import load_model
from .. import config

EMBEDDINGS = load_model(config.embeddings_dir)