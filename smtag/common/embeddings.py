import torch
from .importexport import load_model
from vsearch.net import CatStack
from .. import config

class Embedding:
    def __init__(self, model=None):
        self.model = model

    def __call__(self, x):
        if self.model is not None:
            with torch.no_grad():
                self.model.eval()
                y = self.model(x)
                return y
        else:
            return x

if config.embeddings_model:
    embedding_model = load_model(config.embeddings_model, config.embeddings_dir, CatStack)
else:
    embedding_model = None

EMBEDDINGS = Embedding(embedding_model)