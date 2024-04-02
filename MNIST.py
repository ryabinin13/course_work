#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def data(d):
    train_mnist = pd.read_csv(f'C:/Users/Юрий/Desktop/jupiter/course_work/{d}_train.csv')
    test_mnist = pd.read_csv(f'C:/Users/Юрий/Desktop/jupiter/course_work/{d}_test.csv')
    
    y_train = train_mnist['label'].copy()
    train_mnist = train_mnist.drop('label', axis=1)

    y_test = test_mnist['label'].copy()
    test_mnist = test_mnist.drop('label', axis=1)
    
    scaler = MinMaxScaler()
    train_mnist_norm = scaler.fit_transform(train_mnist)
    test_mnist_norm = scaler.transform(test_mnist)

    train_mnist = pd.DataFrame(train_mnist_norm, columns=train_mnist.columns)*2-1
    test_mnist = pd.DataFrame(test_mnist_norm, columns=test_mnist.columns)*2-1
    
    test_mnist, val_mnist, y_test, y_val = train_test_split(test_mnist, y_test, test_size=0.5)
    
    train_mnist['label'] = y_train
    test_mnist['label'] = y_test
    val_mnist['label'] = y_val
    
    
    return train_mnist, val_mnist, test_mnist


# In[ ]:




