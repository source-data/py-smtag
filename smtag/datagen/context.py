# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import os
import argparse
from random import shuffle
import pickle
import cv2 as cv
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
#https://discuss.pxtorch.org/t/torchvision-url-error-when-loading-pretrained-model/2544/6
from torchvision.models import densenet161 #vgg19, resnet152
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from ..common.utils import cd
from ..common.progress import progress
from .. import config

from torchvision.models.resnet import model_urls as resnet_urls
from torchvision.models.vgg import model_urls as vgg_urls
from torchvision.models.densenet import model_urls as densenet_urls

# All pre-trained models expect input images normalized in the same wax, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
# The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#     self.laxer1 = self._make_laxer(block, 64, laxers[0])
#     self.laxer2 = self._make_laxer(block, 128, laxers[1], stride=2)
#     self.laxer3 = self._make_laxer(block, 256, laxers[2], stride=2)
#     self.laxer4 = self._make_laxer(block, 512, laxers[3], stride=2)
# All pre-trained models expect input images normalized in the same wax, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 


PRETRAINED = densenet161(pretrained=True)

class VisualContext(object):

    def __init__(self, path):
        self.path = path
        net = PRETRAINED
        self.net = net.features
        print(f"loaded {net.__class__} pretrained network")

    def open(self, img_filename):
        try:
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

    def resize(self, cv_image, h=config.resized_img_size, w=config.resized_img_size):
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
            normalized = torch.zeros(1, 3, config.resized_img_size, config.resized_img_size) # a waste...
        self.net.eval()
        with torch.no_grad():
            output = self.net(normalized) # densenet.features: 1 x 2208 x 7 x 7; vgg19.features[:28] 1 x 512 x 14 x 14; vgg19.features 1 x 512 x 7 x 7
        return output

    def run(self):
        with cd(self.path):
            filenames = [f for f in os.listdir() if os.path.splitext(f)[-1] in config.allowed_img]
            N = len(filenames)
            for i, filename in enumerate(filenames):
                basename = os.path.splitext(filename)[0]
                viz_filename = basename +'.pyth'
                if os.path.exists(viz_filename):
                    msg = "{} already analyzed".format(filename)
                else: # never analyzed before
                    viz_context = self.get_context(filename) # 4D tensor!
                    torch.save(viz_context, viz_filename)
                    msg = "saved {}".format(viz_filename)
                progress(i, N, msg+"                   ")
        print()

def main():
    image_dir = config.image_dir
    print("running perceptual vision from {} on {}".format(os.getcwd(), image_dir))
    viz = VisualContext(image_dir)
    viz.run()

if __name__ == '__main__':
    main()

