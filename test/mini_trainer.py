import torch
from torch import nn, optim
from smtag.train.builder import SmtagModel
from smtag.common.progress import progress

def toy_model(x, y, selected_features = ['geneprod'], overlap_features = [], collapsed_features = [], threshold = 1E-02, epochs = 100):
        opt = {}
        opt['namebase'] = 'test_importexport'
        opt['learning_rate'] = 0.01
        opt['epochs'] = epochs
        opt['minibatch_size'] = 1
        opt['selected_features'] = selected_features
        opt['collapsed_features'] = collapsed_features
        opt['overlap_features'] = overlap_features
        opt['nf_table'] =  [8]
        opt['pool_table'] = [2]
        opt['kernel_table'] = [3]
        opt['dropout'] = 0.1
        opt['nf_input'] = x.size(1)
        opt['nf_output'] =  y.size(1)
        opt['skip'] = True
        opt['softmax_mode'] = True
        model = SmtagModel(opt)
        # test if on GPU
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "GPUs available.")
            model = nn.DataParallel(model)
            model.cuda()
        # we do the training loop here instead of using smtag.trainer to avoid the need to prepare minibatches
        loss_fn = nn.CrossEntropyLoss() # nn.BCELoss() #
        optimizer = optim.Adam(model.parameters(), lr = opt['learning_rate'])
        optimizer.zero_grad()
        loss = 1
        i = 0
        # We stop as soon as the model has reasonably converged or if we exceed a max number of iterations
        max_iterations = opt['epochs']
        while loss > threshold and i < max_iterations:
            progress(i, max_iterations)
            y_hat = model(x)
            loss = loss_fn(y_hat, y.argmax(1))
            loss.backward()
            optimizer.step()
            i += 1
        print("Model preparation done! {} iterations reached loss={}".format(i, float(loss)))
        # don't forget to set the state of the model to eval() to avoid Dropout
        model.eval()
        return model
