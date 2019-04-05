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
from torchvision.models import vgg19, resnet152, densenet161
from torchvision import transforms
from tensorboardX import SummaryWriter
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

    def __init__(self, path):
        self.path = path
        # VGG19
        # net = vgg19(pretrained=True)
        # self.net = net.features[:28]
        # DENSENET
        net = densenet161(pretrained=True)
        self.net = net.features
        # # RESNET
        # modules = list(resnet152.children())
        # self.net = nn.Sequential(*modules[:9])
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


class PCA_reducer():
    def __init__(self, k, path):
        self.path = path
        self.k = k
        self.pca_model = None

    def train(self, fraction_images_pca_model=config.fraction_images_pca_model): # path to pre-processed viz context tensors for pca training set
        filenames = [f for f in os.listdir(self.path) if os.path.splitext(f)[-1] == '.pyth']
        shuffle(filenames)
        N = int(len(filenames) * fraction_images_pca_model)
        filenames = filenames[:N]
        t = []
        for i, filename in enumerate(filenames):
            progress(i, len(filenames), f"{filename}                    ")
            t.append(torch.load(os.path.join(self.path, filename)))
        t = torch.cat(t, 0)
        self.pca_model = PCA(n_components=self.k, svd_solver='randomized').fit(self.convert2np(t)) # approximation for large datasets # IncrementalPCA(n_components=self.k, batch_size=self.k * 5).fit(self.convert2np(t)) #
        return t, filenames[:N] # N x C x H x W

    def convert2np(self, x):
        B, C, H, W = x.size()
        x = x - x.mean()
        x = x / x.std()
        x.transpose_(1, 3) # B x W x H x C
        x.resize_(B * W * H, C) # rows=vectorized position, columns = features
        x_np = x.numpy()
        return x_np

    def reduce(self, x, grid_size=config.img_grid_size):
        B, C, H, W = x.size()
        x_np = self.convert2np(x) # B*W*H x C
        p_np = self.pca_model.transform(x_np) # B*W*H x k
        p_th = torch.from_numpy(p_np)
        p_th.resize_(B, W, H, self.k) # B x W x H x k
        p_th.transpose_(1, 3) # B x k x H x W
        # x_reduced = F.adaptive_max_pool2d(p_th, grid_size) # alternative: F.adaptive_avg_pool2d(p_th, grid_size)
        x_reduced = p_th # no pool
        # x_reduced = torch.sigmoid(x_reduced) # alternatives: x_reduced /= x_reduced.max(); or: x_reduced -= x_reduced.mean(); x_reduced /= x_reduced.std();
        x_reduced = (x_reduced - x_reduced.min()) / (x_reduced.max() - x_reduced.min()) # minmax rescaling
        return x_reduced.view(B, self.k*grid_size*grid_size) # 4D B x k * 3 * 3

def main():
    parser = config.create_argument_parser_with_defaults(description='Exracting visual context vectors from images')
    parser.add_argument('-F', '--fraction', type=float, default = config.fraction_images_pca_model, help='Fraction of images to be used to train pca model.')
    parser.add_argument('--pca', action='store_true', default=False, help='Train the PCA model only, without re-extracting featues from images.')

    args = parser.parse_args()
    image_dir = config.image_dir
    fraction_images_pca_model = args.fraction
    pca_only = args.pca    
    if pca_only:
        print("Performing only PCA, without running perceptual vision.")
    else:
        print("running perceptual vision from {} on {}".format(os.getcwd(), image_dir))
        viz = VisualContext(image_dir)
        viz.run()

    pca = PCA_reducer(config.k_pca_components, image_dir)
    print(f"\ntraining pca model on viz context files...")
    trainset, filenames = pca.train(fraction_images_pca_model)
    print("\nDone!")
    pca_reducer_filename = os.path.join(pca.path, "pca_model.pickle")
    with open(pca_reducer_filename, "wb") as f:
        pickle.dump(pca, f)
        print(f"PCA model saved to {pca_reducer_filename}")
    # print("Reducing trainset for visualization...")
    # reduced = pca.reduce(trainset)# print("Done! Writing to tensorboard.")
    # writer = SummaryWriter()
    # writer.add_embedding(trainset.transpose(1, 3).transpose(1, 2).contiguous().view(-1, trainset.size(1)), tag="trainset") # # N x C x H x W --> # N*H*W x C
    # writer.add_embedding(reduced.view(reduced.size(0), pca.k, config.img_grid_size, config.img_grid_size).transpose(1, 3).transpose(1, 2).contiguous().view(-1, pca.k), tag="reduced")

if __name__ == '__main__':
    main()

