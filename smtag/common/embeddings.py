import torch
from torch import nn
from .importexport import load_model
from catstack.model import CatStack, Hyperparameters
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
    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPUs available for embeddings.")
        # device = torch.device("cuda:0")
        embedding_model = embedding_model.cuda()
        gpu_model = nn.DataParallel(embedding_model)
        gpu_model.hp = embedding_model.hp # not sure if necessary
        #gpu_model.to(device)
        embedding_model = gpu_model
else:
    embedding_model = None

EMBEDDINGS = Embedding(embedding_model)