import torch

def replace(x, mask, replacement):

    assert(x.size(0)==replacement.size(0))
    assert(x.size(1)==mask.size(0))

    nf = x.size(0)
    L = x.size(1)
    print(nf, L)

    replacement.resize_(nf, 1)
    replacement_m = torch.cat([replacement for _ in range(L)], 1)

    mask.resize_(1,L)
    mask_inv = mask.clone()
    mask_inv[mask_inv==0] = 2
    mask_inv[mask_inv==1] = 0
    mask_inv[mask_inv==2] = 1
    mask_m = torch.cat([mask for _ in range(nf)], 0)
    mask_inv_m = torch.cat([mask_inv for _ in range(nf)], 0)

    masked_replacement = replacement_m * mask_m

    x_inactivate = x * mask_inv_m
    x_replaced = x_inactivate + masked_replacement

    return x_replaced





