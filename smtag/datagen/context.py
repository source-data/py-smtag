# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import cv2 as cv
import torch
from torch import nn
import torchvision
#https://discuss.pytorch.org/t/torchvision-url-error-when-loading-pretrained-model/2544/6
from torchvision.models import resnet152
from torchvision.models.resnet import model_urls
model_urls['resnet152'] = model_urls['resnet152'].replace('https://', 'http://')
from torchvision import transforms
from ..common.utils import cd


# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
# The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#     self.layer1 = self._make_layer(block, 64, layers[0])
#     self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#     self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#     self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
# https://discuss.pytorch.org/t/how-can-l-use-the-pre-trained-resnet-to-extract-feautres-from-my-own-dataset/9008
# from torch.autograd import Variable
# resnet152 = models.resnet152(pretrained=True)
# modules=list(resnet152.children())[:-1]
# resnet152=nn.Sequential(*modules)
# for p in resnet152.parameters():
#     p.requires_grad = False

# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:


# modules = list(resnet152(pretrained=True).children())[:-1]
# m = nn.Sequential(*modules)
# x = torch.zeros(1, 3, 244, 244)
# m(x).size() # -->torch.Size([1, 2048, 2, 2])
# y.numel() # --> 8192


class VisualContext(object):

    def __init__(self, path, selected_output_module=-1):
        self.path = path
        modules = list(resnet152(pretrained=True).children())[:selected_output_module]
        self.net = nn.Sequential(*modules)
        #self.viz_ctx_features =  viz_ctx_features

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
        BGR = torch.from_numpy(cv_image) # cv image is BGR # cv images are height   x width  x channels
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

    def resize(self, cv_image):
        print("resizing", cv_image.shape)
        resized = cv.resize(cv_image, (224, 224), interpolation=cv.INTER_AREA)
        return resized

    def normalize(self, image):
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return normalizer(image)

    def get_context(self, filename):
        cv_image = self.open(filename)
        if cv_image is not None:
            resized = self.resize(cv_image)
            image = self.cv2th(resized)
            normalized = self.normalize(image)
            normalized.unsqueeze_(0)
        else: 
            normalized = torch.zeros(1, 3, 224, 224) # a waste...
        self.net.eval()
        with torch.no_grad():
            output = self.net(normalized)
        # NORMALIZE OUTPUT TO 0..1 ? 
        n = output.numel() # number of elements per minibatch
        vectorized = output.view(n) # flatten tensor to batch of vectors
        return vectorized # 1D n