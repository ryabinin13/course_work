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

def train_epoch(model, optimizer, criterion, train_loader):
    mean_loss = []
    mean_accuracy = []
    for batch_train in train_loader:
        optimizer.zero_grad()
        pred = model(batch_train['sample'])
        loss = criterion(pred, batch_train['target'])
        loss.backward()
        optimizer.step()
        
        mean_accuracy.extend((torch.argmax(pred, dim=-1) == batch_train['target']).numpy().tolist())
        mean_loss.append(loss)
            
    return mean_loss, mean_accuracy

def train(model, criterion, optimizer, n_epochs, train_loader, val_loader):
    check_loss = [10**5]
    val_loss_log = []
    val_acc_log = []
    for i in range(n_epochs):
        loss_train, acc_train = train_epoch(model, optimizer, criterion, train_loader)

        loss_val, acc_val = test(model, criterion, val_loader)
        val_loss_log.append(np.mean(loss_val))
        val_acc_log.append(np.mean(acc_val))

        print(f'epoch = {i + 1}, loss = {np.mean(loss_val)}, accuracy = {np.mean(acc_val)}')
        check_loss.append(np.mean(loss_val))
        if (check_loss[-1] > check_loss[-2]):
            break   
        if i > 10 and abs(check_loss[-1] - check_loss[-2]) < 0.001:
            break
    return val_loss_log, val_acc_log

def test(model, criterion, load):
    mean_loss = []
    mean_accuracy = []
    for batch in load:
        with torch.no_grad():
            pred = model(batch['sample'])
            loss = criterion(pred, batch['target'])
            
            mean_accuracy.extend((torch.argmax(pred, dim=-1) == batch['target']).numpy().tolist())
            mean_loss.append(loss)
            
    return np.mean(mean_loss), mean_accuracy


# In[ ]:




