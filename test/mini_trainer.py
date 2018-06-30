import torch
from torch import nn, optim
from smtag.builder import SmtagModel
from smtag.progress import progress

def toy_model(x, y, selected_features = ['geneprod'], threshold = 1E-02, epochs = 100):
        opt = {}
        opt['namebase'] = 'test_importexport'
        opt['learning_rate'] = 0.01
        opt['epochs'] = epochs
        opt['minibatch_size'] = 1
        opt['selected_features'] = selected_features
        opt['collapsed_features'] = []
        opt['overlap_features'] = []
        opt['nf_table'] =  [8,8]
        opt['pool_table'] = [2,2]
        opt['kernel_table'] = [2,2]
        opt['dropout'] = 0.1
        opt['nf_input'] = x.size(1)
        opt['nf_output'] =  y.size(1)
        model = SmtagModel(opt)
        # test if on GPU
        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count(), "GPUs available.")
            model = nn.DataParallel(self.model)
            model.cuda()
        # we do the training loop here instead of using smtag.trainer to avoid the need to prepare minibatches
        loss_fn = nn.SmoothL1Loss() # nn.BCELoss() # 
        optimizer = optim.Adam(model.parameters(), lr = opt['learning_rate'])
        optimizer.zero_grad()
        loss = 1
        i = 0
        # We stop as soon as the model has reasonably converged or if we exceed a max number of iterations
        max_iterations = opt['epochs']
        while loss > threshold and i < max_iterations:
            progress(i, max_iterations)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            i += 1
        print("Model preparation done! {} iterations reached loss={}".format(i, float(loss)))
        # don't forget to set the state of the model to eval() to avoid Dropout
        model.eval()
        return model