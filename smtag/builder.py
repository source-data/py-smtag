import torch
from torch import nn
from collections import OrderedDict

class Builder():
    
    def __init__(self, opt):
        self.nf_input = opt['nf_input']
        self.nf_output = opt['nf_output']
        self.nf_table = opt['nf_table']
        self.kernel_table = opt['kernel_table']
        self.pool_table = opt['pool_table']
        self.dropout = opt['dropout']
        self.autoencode = opt['autoencode']
        self.model = self.build()
    
    def build(self):
        pre = nn.BatchNorm2d(self.nf_input)
        core = nn.Sequential(Unet2(self.nf_input, self.nf_table, self.kernel_table, self.pool_table, self.dropout, self.autoencode),
                             nn.Conv2d(self.nf_input, self.nf_output, (1, 1), (1, 1)),
                             nn.BatchNorm2d(self.nf_output)
                            )
        post = nn.Sigmoid()
        return nn.Sequential(pre, core, post)

class Unet2(nn.Module):
    def __init__(self, nf_input, nf_table, kernel_table, pool_table, dropout, autoencode):
        super(Unet2, self).__init__()
        self.level = len(kernel_table)
        self.nf_input = nf_input
        self.nf_table = nf_table
        self.nf_output = nf_table.pop(0)
        self.pool_table = pool_table
        self.pool = pool_table.pop(0)
        self.kernel_table = kernel_table
        self.kernel = kernel_table.pop(0)
        if self.kernel % 2 == 0:
           self.padding = int(self.kernel/2)
        else:
           self.padding = math.floor((self.kernel-1)/2)
        self.dropout = dropout
        self.autoencode = autoencode
        
                
    def forward(self, x):
        y = nn.Dropout(self.dropout)(x)
        y = nn.Conv2d(self.nf_input, self.nf_input, (1, self.kernel), (1, 1), (0, self.padding))(y)
        y = nn.BatchNorm2d(self.nf_input)(y)
        y = nn.ReLU(True)(y)
        y_size_1 = y.size()
        y, pool_1_indices = nn.MaxPool2d((1, self.pool), (1, self.pool), return_indices=True)(y)
        y = nn.Conv2d(self.nf_input, self.nf_output, (1, self.kernel), (1, 1), (0, self.padding))(y)
        y = nn.BatchNorm2d(self.nf_output)(y)
        y = nn.ReLU(True)(y)
        
        if len(self.nf_table) > 0:
            y_size_2 = y.size()
            y, pool_2_indices = nn.MaxPool2d((1, self.pool), (1, self.pool), return_indices=True)(y)
            y = Unet2(self.nf_output, self.nf_table, self.kernel_table, self.pool_table, self.dropout, self.autoencode)(y)
            y = nn.MaxUnpool2d((1, self.pool), (1, self.pool))(y, pool_2_indices, y_size_2)
        
        y = nn.Dropout(self.dropout)(y)
        y = nn.ConvTranspose2d(self.nf_output, self.nf_input, (1, self.kernel), (1, 1), (0, self.padding))(y)
        y = nn.BatchNorm2d(self.nf_input)(y)
        y = nn.ReLU(True)(y)
        y = nn.MaxUnpool2d((1, self.pool), (1, self.pool))(y, pool_1_indices, y_size_1)
        y = nn.ConvTranspose2d(self.nf_input, self.nf_input, (1, self.kernel), (1, 1), (0, self.padding))(y)
        y = nn.BatchNorm2d(self.nf_input)(y)
        y = nn.ReLU(True)(y)
        
        if not self.autoencode:
            y = torch.cat((y, x), 1)
            y = nn.Conv2d(2*self.nf_input, self.nf_input, (1, 1), (1, 1))(y)
            
        return y


    