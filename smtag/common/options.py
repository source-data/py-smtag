from .mapper import Catalogue, concept2index
from toolbox.models import HyperparametersCatStack
from .. import config

# TODO: derive this by extending toolbox.models.Hyperparameters class
class HyperparametersSmtagModel(HyperparametersCatStack):

    def __init__(self, opt=None):
        self.descriptor = "undefined"
        if opt is not None:
            self.descriptor = "; ".join([f"{k}={opt[k]}" for k in opt])
            self.namebase = opt['namebase']
            self.data_path_list = opt['data_path_list']
            self.modelname = opt['modelname']
            self.learning_rate = opt['learning_rate']
            self.epochs = opt['epochs']
            self.minibatch_size = opt['minibatch_size']
            self.L = None # can only be update when loading dataset...
            self.nf_input = opt['nf_input']
            self.selected_features = Catalogue.from_list(opt['selected_features'])
            self.nf_output = len(self.selected_features)
            self.hidden_channels = opt['hidden_channels']
            self.dropout_rate = opt['dropout_rate']
            self.N_layers = opt['N_layers']
            self.kernel = opt['kernel']
            self.padding = opt['padding']
            self.stride = opt['stride']
            
            # softmax requires an <untagged> class
            self.index_of_notag_class = self.nf_output
            self.nf_output += 1

            print("nf.output=", self.nf_output)
            print("nf.input=", self.nf_input)
            print("concept2index self.selected_features", [concept2index[f] for f in self.selected_features])

    def __str__(self):
        return self.descriptor

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     return state
    
    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    # def state_dict(self):
    #     return self.__getstate__()

    # def load_state_dict(self, state):
    #     self.__setstate__(state)
