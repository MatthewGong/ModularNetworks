from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data.sampler import Sampler,SubsetRandomSampler
import torchvision

import os
import sys

import ast
import numpy as np
import pandas as pd

from sqlalchemy import Column, ForeignKey, Integer, String, Table
from sklearn.metrics import confusion_matrix
import customtorchsummary as torchsummary

def train(dataloaded,verbose=True):

    ### ENVIRONMENT ###

    # Hardware compatibility checks
    CUDA = torch.cuda.is_available()

    DEVICE = "cuda" if CUDA else "cpu"
    
    # manage print functions
    v_print = print if verbose else lambda *a, **k: None

    ### /ENVIROMENT ###
    
    ### DATASET CONSTRUCTION ###

    
    ## Augmentation ##
    

    ## Train Test Split ##

    ### /DATASET CONSTRUCTION ###
     
    ### BUILD NETWORK ### 
    
    
    neuralnetwork.to_device(device)
    
    if verbose:
        torchsummary.summary(full_net,input_size=[[(1,900)],[(1,118)]])

    ### /BUILD NETWORK ###

    ### TRAINING PARAMS ###
    
    BATCH = 100
    LR=.01
    EPOCHS = 50

    v_print("Batches: {}\nLearning Rate: {}\nEpochs: {}".format(BATCH,LR,EPOCHS))

    loss_func = nn.CrossEntropyLoss().to_device(cuda)
    optimizer = torch.optim.SGD(neural.parameters(), lr=LR, momentum=0.9)

    ### /TRAINING PARAMS ###


    ### Training Loop ###
    for j in range(EPOCHS):

        running_loss = 0.0
        counter = 0
        
        for step, data in enumerate(train_loaded):
            
            ## forward pass ##
            output = neuralnetwork()

            ## calculate loss ##
            loss = loss_func()
            
            ## gradient update ##
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            ## record keeping and updates ##
            counter += 1
            running_loss += loss.item()

            if counter % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (j+1, counter + 1, running_loss /10))
                running_loss = 0.0
 

    ### Save Network ###
    torch.save(neuralnetwork, savename + ".pt")
