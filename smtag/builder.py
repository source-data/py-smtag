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
        self.model = self.build()
    
    def build(self):
        pre = nn.BatchNorm2d(self.nf_input)
        core = nn.Sequential(Unet2(self.nf_input, self.nf_table, self.kernel_table, self.pool_table, self.dropout),
                             nn.Conv2d(self.nf_input, self.nf_output, (1, 1), (1, 1)),
                             nn.BatchNorm2d(self.nf_output)
                            )
        post = nn.Sigmoid()
        return nn.Sequential(pre, core, post)

class Unet2(nn.Module):
    def __init__(self, nf_input, nf_table, kernel_table, pool_table, dropout):
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
        self.dropout_rate = dropout
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.conv_down_A = nn.Sequential(
            nn.Conv2d(self.nf_input, self.nf_input, (1, self.kernel), (1, 1), (0, self.padding)),
            nn.BatchNorm2d(self.nf_input),
            nn.ReLU(True)
        )
        self.conv_down_B = nn.Sequential(
            nn.Conv2d(self.nf_input, self.nf_output, (1, self.kernel), (1, 1), (0, self.padding)),
            nn.BatchNorm2d(self.nf_output),
            nn.ReLU(True)
        )
        self.conv_up_B = nn.Sequential(
            nn.ConvTranspose2d(self.nf_output, self.nf_input, (1, self.kernel), (1, 1), (0, self.padding)),
            nn.BatchNorm2d(self.nf_input),
            nn.ReLU(True),
        )
        self.conv_up_A = nn.Sequential(
            nn.ConvTranspose2d(self.nf_input, self.nf_input, (1, self.kernel), (1, 1), (0, self.padding)),
            nn.BatchNorm2d(self.nf_input),
            nn.ReLU(True)
        )
        if len(self.nf_table) > 0:
            self.unet2 = Unet2(self.nf_output, self.nf_table, self.kernel_table, self.pool_table, self.dropout_rate)
        else:
            self.unet2 = None
        self.reduce = nn.Conv2d(2*self.nf_input, self.nf_input, 1, 1)
                
    def forward(self, x):
        y = self.dropout(x)
        y = self.conv_down_A(y)
        y_size_1 = y.size()
        y, pool_1_indices = nn.MaxPool2d((1, self.pool), (1, self.pool), return_indices=True)(y)
        y = self.conv_down_B(y)
        
        if self.unet2 is not None:
            y_size_2 = y.size()
            y, pool_2_indices = nn.MaxPool2d((1, self.pool), (1, self.pool), return_indices=True)(y)
            y = self.unet2(y)
            y = nn.MaxUnpool2d((1, self.pool), (1, self.pool))(y, pool_2_indices, y_size_2)
        
        y = self.dropout(y)
        y = self.conv_up_B(y)
        y = nn.MaxUnpool2d((1, self.pool), (1, self.pool))(y, pool_1_indices, y_size_1)
        y = self.conv_up_A(y)
        
        y = x + y # residual block way
        #y = torch.cat((x, y), 1) # original way of doing the shortcut; seems not not work anymore :-(
        #y = self.reduce(y) 
            
        return y


    