import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data.sampler import Sampler,SubsetRandomSampler
import torchvision

#from torchsummary import summary

import os
import sys

import seaborn as sn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ast
import numpy as np
import pandas as pd

from sqlalchemy import Column, ForeignKey, Integer, String, Table
from sklearn.metrics import confusion_matrix
import customtorchsummary as torchsummary

import ModularNetGenerator as Nets
import NetManufacturing as Factory
import ExperimentalFunctions as EF
import DatasetAugmentation as Augmentation


def build_train(BY,URL,REL,BATCH,FOLDS,LR,num_layers,EPOCHS,name):
    
    """
    Loads and augments a dataset based on inputs
    """

    ### Loads dataset
    data_query = EF.query_dataset(URL,BY,REL)

    ### Construct the dataset
    d_stack,c_stack,l_stack = EF.build_dataset(data_query,'d',BY)

    ### Make weights if flagged
    weights, num_classes= EF.make_weights(l_stack)

    ### train test split
    folds_indicies = np.array_split(np.random.permutation(len(l_stack)),FOLDS)

    ### Augments dataset
    Diff_Aug = Augmentation.ShiftNoise()
    Chem_Aug = Augmentation.PercentNoise()
    
    ### Cross validation vs Train Validate Test   
    for k in range(FOLDS):

        ### Cross validation splitting
        mask = [m for m in range(FOLDS) if m!= k]
        
        test_data = Data.Subset(all_data,folds_indicies[k])

        chunked =np.concatenate([folds_indicies[f] for f in mask])
        
        train_data = Data.Subset(all_data,chunked)
        
        # fold_sampler makes it so it's already shuffled
        test_loaded = Data.DataLoader(dataset=test_data,batch_size=BATCH,shuffle=True)
        train_loaded = Data.DataLoader(dataset=train_data,batch_size=BATCH,shuffle=True)

        vprint(len(test_loaded.dataset),len(train_loaded.dataset))
        
        for i in range(1):
        
            ### TRAINING PARAMS
        
            ### END TRAINING PARAMS

            ### Network
            """
            Network should be fed in as a parameter for easier debugging
            """

            temp_name = name + str(k)
            
            ### Determine loss function
            """
             Check network flags, otherwise infer from task
            """

            loss_func = nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda())
            optimizer = torch.optim.SGD(full_net.parameters(), lr=LR, momentum=0.9)

            ### Training Loop
            """
            loop per epoch and train network
            """
            for j in range(EPOCHS):
                running_loss = 0.0
                counter = 0
                for step,(*args) in enumerate(train_loaded):
                    
                    output = full_net(*args)

                    loss = loss_func(*args)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # record keeping and updates
                    counter += 1
                    running_loss += loss.item()

                    if counter % 10 == 9:    # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (j+1, counter + 1, running_loss /10))
                        running_loss = 0.0
                        
            # Save the net for future evaluation
            torch.save(full_net,temp_name+ ".pt")
            

            