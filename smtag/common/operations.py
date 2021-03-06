# -*- coding: utf-8 -*-
import torch
from torch import nn

def t_replace(x, mask, replacement):
    '''
    A function to replace the columns that are marked by a mask by a replacement column.
    Example with
    mask:        0110 the second and third columns of x need to be replaced

    x:           1011   replacement column: 1
                 1010                       1
                 0100                       1

    result:      1111 the second and third columns have been replaced by the replacement colum
                 1110
                 0110

    With x as 3D tensor, first dimension is the example index, the second is the row index and third is the column index.

    Args:
        x (torch.Tensor): a 3D N x nf x L Tensor filled with 0 or 1 values, where N is number of examples with L columns, nf number of features in a column, L is length of a row or number of columns.
        mask (torch.Tensor): a 2D N x L Tensor, filled with 0 or 1 values. For each example, the columns to be changes in x (second dimension) are labeled with 1 in mask.
        replacement (torch.Tensor): a 3D 1 x nf x 1 Tensor filled with 0 or 1 values. Each column of x labeled by mask will be replaced by replacement.

    Returns:
       (torch.Tensor): a 3D tensor where the columns labeled by the mask have been replaced by the replacement column.
    '''

    # replace this with x.index_copy_(3, mask3D, replacement3D) or somethign similar: nope, index cannot be 3D
    # maybe with something like x[:,:][mask] = replacement_vector: nope ;-()

    assert(x.size(1)==replacement.size(1))
    assert(x.size(2)==mask.size(1))

    N, nf, L = x.size()

    replacement_3D = replacement.repeat(N, 1, L) # the replacement colum is replicated through all L columns and N examples


    mask_inv = 1 - mask # 0s become 1s and 1s become 0s
    mask_3D = mask.unsqueeze(1).repeat(1, nf, 1) # the mask is replicated through each nf rows
    mask_inv_3D = mask_inv.unsqueeze(1).repeat(1, nf, 1) # the mask is replicated through each nf rows

    masked_replacement = replacement_3D * mask_3D

    x_inactivate = x * mask_inv_3D
    x_replaced = x_inactivate + masked_replacement

    return x_replaced

