
import torch
from torch import nn
from .. import config

def attn(a, b): # we could save 2 transpose operations if b is provided directly as B_b, H_b, N, L
    # input need to be in format Batch x Heads x Length x Channel
    B_a, H_a, R, N_a = a.size()
    B_b, H_b, L, N_b = b.size() 
    assert B_a == B_b, "mismatch batch size for a ({B_a}) and b ({B_b}) in attention layer."
    assert H_a == H_b, "mismatch attention heads for a ({H_a}) and b ({H_b}) in attention layer."
    assert N_a == N_b, "mismatch length (width) for a ({N_a}) and b ({N_b}) in attention layer."
    position_wise_interactions = torch.matmul(a, b.transpose(2, 3).contiguous()) # RxN * NxL -> RxL
    weights = torch.softmax(position_wise_interactions / sqrt(N_a), -2) # RxL
    attention = torch.matmul(a.transpose(2, 3).contiguous(), weights) # NxR * RxL -> NxL
    return attention


class Hyperparameters:

    def __init__(
        self,
        attn_on = True,
        in_channels = config.nbits,
        out_channels = config.embedding_out_channels,
        nf_table      = [128,128,128,128,256,256,256,256,512,512],
        attn_heads =    [  2,  2,  2,  2,  4,  4,  4,  4,  8,  8],
        kernel_table  = [  7] * 10,
        padding_table = [  3] * 10,
        stride_table =  [  1] * 10,
        dropout_rate = 0.2,
    ):

        self.attn_on = attn_on
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nf_table = nf_table
        self.kernel_table  = kernel_table
        self.padding_table = padding_table
        self.stride_table =  stride_table
        self.dropout_rate = dropout_rate
        self.attn_heads = attn_heads

    def __str__(self):
        return f'\
attn_{self.attn_on}_\
nf{"_".join([str(x) for x in self.nf_table])}_\
k{"".join([str(x) for x in self.kernel_table])}_\
p{"".join([str(x) for x in self.padding_table])}_\
s{"".join([str(x) for x in self.stride_table])}_\
d{str(self.dropout_rate).replace(".", "")}'

    def __getstate__(self):
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

    def state_dict(self):
        return self.__getstate__()

    def load_state_dict(self, state):
        self.__setstate__(state)


HP = Hyperparameters()

class CatStack(nn.Module):

    def __init__(self, hp=HP) -> torch.Tensor:
        super(CatStack, self).__init__()
        self.hp = hp
        self.N_layers = len(self.hp.nf_table)
        self.BN_in = nn.BatchNorm1d(self.hp.in_channels)
        self.blocks = nn.ModuleList()
        self.poswise_linears = nn.ModuleList()
        in_channels = self.hp.in_channels
        self.out_channels = self.hp.out_channels
        cumul_channels = in_channels # features of original input included
        for i in range(self.N_layers):
            out_channels = self.hp.nf_table[i]
            cumul_channels += out_channels
            self.poswise_linears.append(nn.Conv1d(in_channels, in_channels, 1, 1))
            if self.hp.attn_on and self.hp.attn_heads[i] > 0:
                in_channels *= 2 # because the output of self-attention is concatenated with original input
            block = nn.Sequential(
                nn.Dropout(self.hp.dropout_rate),
                nn.Conv1d(in_channels, out_channels, self.hp.kernel_table[i], self.hp.stride_table[i], self.hp.padding_table[i]),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(out_channels),
                # nn.MaxPool1d(2, 2),
            )
            self.blocks.append(block)
            in_channels = out_channels
        self.compress = nn.Conv1d(cumul_channels, self.out_channels, 1, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.BN_in(x)
        x_list = [x.clone()]
        for i in range(self.N_layers):
            if self.hp.attn_on and self.hp.attn_heads[i] > 0: # this allows to switch off/on attention in specific layers
                # multi-head attention
                B, C, L = x.size()
                x_h = self.poswise_linears[i](x)
                x_h = x_h.view(B, self.hp.attn_heads[i], C // self.hp.attn_heads[i], L)
                att = attn(x_h.transpose(2, 3).contiguous(), x_h.transpose(2, 3).contiguous())
                att = att.view_as(x)
                x = torch.cat((att, x), 1)
            x = self.blocks[i](x)
            # x = F.interpolate(x, config.max_length, mode='linear')
            x_list.append(x.clone())
        y = torch.cat(x_list, 1)
        y = self.compress(y)
        return y
