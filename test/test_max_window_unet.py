import torch
import argparse
from smtag.train.builder import Unet2
from smtag.common.progress import progress

from smtag import config

NBITS = config.nbits

def find_min_length(D=4, k=6, p=2, L=2000):
    min_length = []
    for n in range(D):
        d = n+1
        U = Unet2(NBITS, [1]*d, [k]*d, [p]*d, 0.1)
        print(U, L)
        m = test_model(U, L)
        min_length.append((d,m))
        print(f"\nUnet2 with depth {d} can accept minimal length: {m+1}")
    return min_length

def test_model(U, L=2000):
    for l in range(L, 0, -1):
        progress(L-l-1, L, f"{l}")
        x = torch.zeros(1, NBITS, l)
        try:
            U(x)
        except Exception as e:
            print("\n", e) 
            if l == L:
                min_length = ">{}".format(l)
            else:
                min_length = l
            break
        else:
            if l == 1:
                min_length = l
    return min_length


def main():
    parser = argparse.ArgumentParser(description='Little utility to find out what is the miminum text size for a given Unet2.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-k', '--kernel', type=int, default=6, help='kernel width')
    parser.add_argument('-p', '--pool', type=int, default=2, help='pooling window')
    parser.add_argument('-L', '--max_length', type=int, default=1200, help='max length to be tested')
    parser.add_argument('-D', '--max_depth', type=int, default=4, help='maximal depth of unet')

    arguments = parser.parse_args()
    k = arguments.kernel
    p = arguments.pool
    L = arguments.max_length
    D = arguments.max_depth
    min_length = find_min_length(D, k, p, L)
    print("\ndepth\tmax_window_size")
    print("\n".join(["{}\t{}".format(d,m) for d,m in min_length]))

if __name__ == "__main__":
    main()