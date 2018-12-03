# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import os
import argparse
from random import shuffle
import cv2 as cv
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
#https://discuss.pxtorch.org/t/torchvision-url-error-when-loading-pretrained-model/2544/6
from torchvision.models import vgg19, resnet152
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA
from ..common.utils import cd
from ..common.progress import progress

from torchvision.models.resnet import model_urls as resnet_urls
from torchvision.models.vgg import model_urls as vgg_urls
for m in resnet_urls:
    resnet_urls[m] = resnet_urls[m].replace('https://', 'http://')
for m in vgg_urls:
    vgg_urls[m] = vgg_urls[m].replace('https://', 'http://')


# All pre-trained models expect input images normalized in the same wax, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
# The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#     self.laxer1 = self._make_laxer(block, 64, laxers[0])
#     self.laxer2 = self._make_laxer(block, 128, laxers[1], stride=2)
#     self.laxer3 = self._make_laxer(block, 256, laxers[2], stride=2)
#     self.laxer4 = self._make_laxer(block, 512, laxers[3], stride=2)

#model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
# https://discuss.pxtorch.org/t/how-can-l-use-the-pre-trained-resnet-to-extract-feautres-from-mx-own-dataset/9008
# from torch.autograd import Variable
# resnet152 = models.resnet152(pretrained=True)
# modules=list(resnet152.children())[:-1]
# resnet152=nn.Sequential(*modules)
# for p in resnet152.parameters():
#     p.requires_grad = False

# All pre-trained models expect input images normalized in the same wax, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:


# modules = list(resnet152(pretrained=True).children())[:-1]
# m = nn.Sequential(*modules)
# x = torch.zeros(1, 3, 244, 244)
# m(x).size() # -->torch.Size([1, 2048, 2, 2])
# x.numel() # --> 8192


class VisualContext(object):

    def __init__(self, path, selected_output_module=28):
        self.path = path
        modules = list(vgg19(pretrained=True).features) # children() for resnet
        self.net = nn.Sequential(*modules[:selected_output_module])

    def open(self, img_filename):
        try:
            with cd(self.path):
                cv_image = cv.imread(img_filename) #H x W x C, BGR
        except Exception as e:
            print("{} not loaded".format(img_filename), e)
            cv_image = None
        return cv_image

    @staticmethod
    def cv2th(cv_image):
        BGR = torch.from_numpy(cv_image) # cv image is BGR # cv images are height x width  x channels
        blu  = BGR[ : , : , 0]
        gre  = BGR[ : , : , 1]
        red  = BGR[ : , : , 2]
        RGB = torch.Tensor(BGR.size())
        RGB[ : , : , 0] = red
        RGB[ : , : , 1] = gre
        RGB[ : , : , 2] = blu
        RGB = torch.transpose(RGB, 2, 0) # transpose to channels x width  x height 
        RGB = torch.transpose(RGB, 1, 2) # transpose to channels x height x width
        RGB = RGB.float() / 255.0
        return RGB

    def resize(self, cv_image, h=224, w=224): # h and w should be specified in config
        resized = cv.resize(cv_image, (h, w), interpolation=cv.INTER_AREA)
        return resized

    def normalize(self, image):
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return normalizer(image)

    def get_context(self, filename):
        cv_image = self.open(filename) # H x W x C
        if cv_image is not None:
            resized = self.resize(cv_image) # 224 x 224
            image = self.cv2th(resized) # 3D C x H x W
            normalized = self.normalize(image)
            normalized.unsqueeze_(0) # 4D 1 x C x H x W
        else: 
            normalized = torch.zeros(1, 3, 224, 224) # a waste...
        self.net.eval()
        with torch.no_grad():
            output = self.net(normalized)
        return output # 4D 1 x 512 x 14 x 14

class PCA_reducer():
    def __init__(self, k, path, N):
        filenames = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.jpg']
        print("{} image files available".format(len(filenames)))
        shuffle(filenames)
        viz = VisualContext(path=path, selected_output_module=-1)
        _, C, H, W = viz.net(torch.zeros(1, 3, 224, 224)).size()
        print("visual context has dimensions (C x H x W):", " x ".join([str(C), str(H), str(W)]))
        t = torch.Tensor(N, C, H, W)
        print("processing {} image files to build fitting dataset".format(N))
        for i in range(N):
            progress(i, N, filenames[i])
            t[i] = viz.get_context(filenames[i])
            # t[i] = torch.load('viz_context.pyth') # instead of passing images every time through vgg
        print("fitting PCA model")
        self.k = k
        self.pca = PCA(n_components=self.k, svd_solver='randomized').fit(self.convert2np(t)) # approximation for large datasets

    def convert2np(self, x):
        B, C, H, W = x.size()
        x = x - x.mean()
        x = x / x.std()
        x.transpose_(1, 3) # B x W x H x C
        x.resize_(B * W * H, C) # rows=vectorized position, columns = features
        x_np = x.numpy()
        return x_np

    def reduce(self, x):
        B, C, H, W = x.size()
        x_np = self.convert2np(x) # B*W*H x C
        p_np = self.pca.transform(x_np) # B*W*H x k
        p_th = torch.from_numpy(p_np)
        p_th.resize_(B, W, H, self.k) # B x W x H x k
        p_th.transpose_(1, 3) # B x k x H x W
        print("reducing resolution by adaptive max pool")
        x_reduced = F.adaptive_max_pool2d(p_th, 3)
        return x_reduced # 4D B x k x 3 x3

def main():
    parser = argparse.ArgumentParser(description='Exracting visual context vectors from images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--dir', default='img_test', help='Path to image directory')
    parser.add_argument('-k' , '--k_components',  default=8, type=int, help='Number of components of the reduced dataset')
    parser.add_argument('-N', '--max_file', type=int, default=100, help='Maximum number of images to analyze')
    
    arguments = parser.parse_args()
    path = arguments.dir
    k = arguments.k_components
    N = arguments.max_file
    p = PCA_reducer(k, path, N)
    print("explained variance:")
    print(p.pca.explained_variance_ratio_)
    print("sum", p.pca.explained_variance_ratio_.sum())

if __name__ == '__main__':
    main()

