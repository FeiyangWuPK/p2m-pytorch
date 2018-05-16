from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.datasets
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F


BATCH_SIZE = 100
LEARNING_RATE = 0.01
EPOCH = 5

transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

trainData = torchvision.datasets.ImageFolder(
    '../data/train', transform)
testData = torchvision.datasets.ImageFolder('../data/test', transform)

trainLoader = torch.utils.data.DataLoader(
    dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(
    dataset=testData, batch_size=BATCH_SIZE, shuffle=False)


class VGG16(nn.Module):
  def __init__(self):
    super(VGG16, self).__init__()
    self.layer1 = nn.Sequential(

        # 1-1 conv layer
        nn.Conv3d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm3d(64),
        nn.ReLU(),

        # 1-2 conv layer
        nn.Conv3d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm3d(64),
        nn.ReLU(),

        # 1 Pooling layer
        nn.MaxPool3d(kernel_size=2, stride=2))

    self.layer2 = nn.Sequential(

        # 2-1 conv layer
        nn.Conv3d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm3d(128),
        nn.ReLU(),

        # 2-2 conv layer
        nn.Conv3d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm3d(128),
        nn.ReLU(),

        # 2 Pooling lyaer
        nn.MaxPool3d(kernel_size=2, stride=2))

    self.layer3 = nn.Sequential(

        # 3-1 conv layer
        nn.Conv3d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm3d(256),
        nn.ReLU(),

        # 3-2 conv layer
        nn.Conv3d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm3d(256),
        nn.ReLU(),

        # 3 Pooling layer
        nn.MaxPool3d(kernel_size=2, stride=2))

    self.layer4 = nn.Sequential(

        # 4-1 conv layer
        nn.Conv3d(256, 512, kernel_size=3, padding=1),
        nn.BatchNorm3d(512),
        nn.ReLU(),

        # 4-2 conv layer
        nn.Conv3d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm3d(512),
        nn.ReLU(),

        # 4 Pooling layer
        nn.MaxPool3d(kernel_size=2, stride=2))

    self.layer5 = nn.Sequential(

        # 5-1 conv layer
        nn.Conv3d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm3d(512),
        nn.ReLU(),

        # 5-2 conv layer
        nn.Conv3d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm3d(512),
        nn.ReLU(),

        # 5 Pooling layer
        nn.MaxPool3d(kernel_size=2, stride=2))

    self.layer6 = nn.Sequential(

        # 6 Fully connected layer
        # Dropout layer omitted since batch normalization is used.
        nn.Linear(4096, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU())

    self.layer7 = nn.Sequential(

        # 7 Fully connected layer
        # Dropout layer omitted since batch normalization is used.
        nn.Linear(4096, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU())

    self.layer8 = nn.Sequential(

        # 8 output layer
        nn.Linear(4096, 1000),
        nn.BatchNorm1d(1000),
        nn.Softmax())

    def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = self.layer3(out)
      out = self.layer4(out)
      out = self.layer5(out)
      vgg16_features = out.view(out.size(0), -1)
      out = self.layer6(vgg16_features)
      out = self.layer7(out)
      out = self.layer8(out)

      return vgg16_features, out


class PercetualPooling(nn.Module):
    def __init__ (self):
        super(PercetualPooling, self).__init__()
        //todo: adding percetual pooling layer
            
class GraphProjection(nn.Module):
    def __init__(self):
        super(GraphProjection, self).__init__()
        //todo:projecting 3d objects to 2d graphs
        
