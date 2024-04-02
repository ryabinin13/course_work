#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MnistConvolutional_1(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(  # 28 x 28
            nn.Conv2d(in_channels=image_channels, out_channels=6, 
                      kernel_size=5, padding='same'),  # 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 x 14
            nn.Conv2d(in_channels=6, out_channels=16, 
                      kernel_size=5, padding='same'),  # 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7 x 7
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=5, padding='same')  # 3 x 3
        )
        
        self.head = nn.Sequential(
            nn.Linear(in_features=490, out_features=50), # 90
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        out = self.head(out)
        return out

############################################################

class MnistConvolutional_2(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(  # 28 x 28
            nn.Conv2d(in_channels=image_channels, out_channels=35, 
                      kernel_size=3, padding='same'),  # 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 x 14
            nn.Conv2d(in_channels=35, out_channels=38, 
                      kernel_size=3, padding='same'),  # 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7 x 7
            nn.Conv2d(in_channels=38, out_channels=10, kernel_size=5, padding='same')  # 3 x 3
        )
        
        self.head = nn.Sequential(
            nn.Linear(in_features=490, out_features=50), # 90
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        out = self.head(out)
        return out
    
###########################################################


class MnistConvolutional_3(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(  # 28 x 28
            nn.Conv2d(in_channels=image_channels, out_channels=3, 
                      kernel_size=7, padding='same'),  # 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 x 14
            nn.Conv2d(in_channels=3, out_channels=4, 
                      kernel_size=7, padding='same'),  # 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7 x 7
            nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, padding='same')  # 3 x 3
        )
        
        self.head = nn.Sequential(
            nn.Linear(in_features=490, out_features=50), # 90
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        out = self.head(out)
        return out


###########################################################


class MnistConvolutional_4(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(  # 28 x 28
            nn.Conv2d(in_channels=image_channels, out_channels=3, 
                      kernel_size=9, padding='same'),  # 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 x 14
            nn.Conv2d(in_channels=3, out_channels=3, 
                      kernel_size=9, padding='same'),  # 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7 x 7
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, padding='same')  # 3 x 3
        )
        
        self.head = nn.Sequential(
            nn.Linear(in_features=490, out_features=50), # 90
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        out = self.head(out)
        return out
    
    
###########################################################


class MnistConvolutional_5(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(  # 28 x 28
            nn.Conv2d(in_channels=image_channels, out_channels=1, 
                      kernel_size=11, padding='same'),  # 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 x 14
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=11, padding='same'),  # 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7 x 7
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding='same')  # 3 x 3
        )
        
        self.head = nn.Sequential(
            nn.Linear(in_features=490, out_features=50), # 90
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        out = self.head(out)
        return out
    
##############################################################

class MnistConvolutional_11(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(  # 28 x 28
            nn.Conv2d(in_channels=image_channels, out_channels=32, 
                      kernel_size=5),  # 28 x 28
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(in_features=18432, out_features=50), # 90
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        out = self.head(out)
        return out
##############################################################

class MnistConvolutional_12(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(  # 28 x 28
            nn.Conv2d(in_channels=image_channels, out_channels=16, 
                      kernel_size=5),  # 24
            nn.ReLU(),
            #nn.MaxPool2d(2),  # 12
            nn.Conv2d(in_channels=16, out_channels=16, 
                      kernel_size=5),  # 20
        )
        self.head = nn.Sequential(
            nn.Linear(in_features=6400, out_features=50), # 90
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        out = self.head(out)
        return out

    
##############################################################

class MnistConvolutional_13(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(  # 28 x 28
            nn.Conv2d(in_channels=image_channels, out_channels=8, 
                      kernel_size=5),  # 24
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, 
                      kernel_size=5),  # 20
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, 
                      kernel_size=5),  # 16
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, 
                      kernel_size=5),  # 12
        )
        self.head = nn.Sequential(
            nn.Linear(in_features=1152, out_features=50), # 90
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        out = self.head(out)
        return out
    
##############################################################

class MnistConvolutional_14(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(  # 28 x 28
            nn.Conv2d(in_channels=image_channels, out_channels=4, 
                      kernel_size=5),  # 24
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, 
                      kernel_size=5),  # 20
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, 
                      kernel_size=5),  # 16
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, 
                      kernel_size=5),  # 12
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, 
                      kernel_size=5),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, 
                      kernel_size=5, padding='same'),  # 8
        )
        self.head = nn.Sequential(
            nn.Linear(in_features=256, out_features=50), # 90
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        out = self.head(out)
        return out
    
    
##############################################################
class MnistConvolutional_15(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(  # 28 x 28
            nn.Conv2d(in_channels=image_channels, out_channels=2, 
                      kernel_size=5),  # 24
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5),  # 20
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5),  # 16
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5),  # 12
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=5, padding='same'),  # 8
        )
        self.head = nn.Sequential(
            nn.Linear(in_features=128, out_features=50), # 90
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        out = self.head(out)
        return out
    
    
##############################################################
class MnistConvolutional_16(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(  # 28 x 28
            nn.Conv2d(in_channels=image_channels, out_channels=1, 
                      kernel_size=5),  # 24
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5),  # 20
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5),  # 16
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5),  # 12
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, 
                      kernel_size=5, padding='same'),  # 8
        )
        self.head = nn.Sequential(
            nn.Linear(in_features=64, out_features=50), # 90
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        out = self.head(out)
        return out