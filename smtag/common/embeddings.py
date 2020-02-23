import os
import torch
from torch import nn
from toolbox.importexport import load_model
from toolbox.models import Container1d, CatStack1d, HyperparametersCatStack
from typing import ClassVar
from .. import config


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class EmbeddingSubModuleNoFoundError(Error):
    """
    Exception raised when the submodule for generatig embeddings is not found in the containner model.
    """

    def __init__(self, sub_module_class: ClassVar):
        super().__init__(f"Could not find {sub_module_class.__name__} submodule. No embedding module available.")


class Embedding:
    def __init__(self, model: nn.Module=None):
        self.model = model

    def __call__(self, x):
        if self.model is not None:
            with torch.no_grad():
                self.model.eval()
                y = self.model(x)
                return y
        else:
            return x

def load_embedding(path):
    embedding_model = None
    container = load_model(path, Container1d, CatStack1d)
    for sub_module in container.children():
        if isinstance(sub_module, CatStack1d):
            embedding_model = sub_module
            break
    if embedding_model is None:
        raise EmbeddingSubModuleNoFoundError(CatStack1d)
    if torch.cuda.is_available() and embedding_model is not None:
        print(f"{torch.cuda.device_count()} GPUs available for embeddings.")
        gpu_model = embedding_model.cuda()
        gpu_model = nn.DataParallel(gpu_model)
        gpu_model.hp = embedding_model.hp
        embedding_model = gpu_model
    return embedding_model

embedding_model = None
if config.embeddings_model:
    path = os.path.join(config.embeddings_dir, config.embeddings_model)
    embedding_model = load_embedding(path)

EMBEDDINGS = Embedding(embedding_model)