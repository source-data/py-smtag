import torch
from smtag.train.builder import Unet2

min_size = []
for n in range(1, 10):
    U = Unet2(32, [8]*n, [6]*n, [2]*n, 0.1)
    for L in range(1200, 1, -1):
        x = torch.zeros(1, 32, L)
        try:
            y = U(x)
        except Exception as e:
            print(n, L, e)
            min_size.append((n, L))
            break

print(min_size)
