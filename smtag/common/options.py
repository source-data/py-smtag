from .mapper import Catalogue, concept2index
from .. import config


class Options():

    def __init__(self, opt=None):
        self.descriptor = "undefined"
        if opt is not None:
            self.descriptor = "; ".join([f"{k}={opt[k]}" for k in opt])
            self.namebase = opt['namebase']
            self.data_path_list = opt['data_path_list']
            self.modelname = opt['modelname']
            self.learning_rate = opt['learning_rate']
            self.dropout = opt['dropout']
            self.skip = opt['skip']
            self.epochs = opt['epochs']
            self.minibatch_size = opt['minibatch_size']
            self.L = None # can only be update when loading dataset...
            self.nf_table = opt['nf_table']
            self.pool_table = opt['pool_table']
            #self.padding_table = opt['padding_table']
            self.kernel_table = opt['kernel_table']
            self.selected_features = Catalogue.from_list(opt['selected_features'])
            self.use_ocr_context = opt['use_ocr_context']
            self.viz_context_table = opt['viz_context_table'] 
            self.nf_input = opt['nf_input']
            if self.use_ocr_context == 'ocr1':
                self.nf_ocr_context = 1 # fusing horizontal and vertial into single detected-on-image feature
            elif self.use_ocr_context == 'ocr2':
                self.nf_ocr_context = 2 # restricting to horizontal / vertical features, disrigarding position
            elif self.use_ocr_context == 'ocrxy':
                self.nf_ocr_context = config.img_grid_size ** 2 + 2 # 1-hot encoded position on the grid + 2 orientation-dependent features
            else:
                self.nf_ocr_context = 0
            if self.use_ocr_context:
                self.nf_input += self.nf_ocr_context
            self.nf_output = len(self.selected_features)
            # softmax requires an <untagged> class
            self.index_of_notag_class = self.nf_output
            self.nf_output += 1

            print("nf.output=", self.nf_output)
            print("nf.input=", self.nf_input)
            print("concept2index self.selected_features", [concept2index[f] for f in self.selected_features])

    def __str__(self):
        return self.descriptor

    def __getstate__(self):
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

    def state_dict(self):
        return self.__getstate__()

    def load_state_dict(self, state):
        self.__setstate__(state)
