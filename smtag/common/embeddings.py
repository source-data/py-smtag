from .importexport import load_model
from .. import config

autoencoder = load_model(config.embeddings_dir)
EMBEDDINGS = autoencoder.embed