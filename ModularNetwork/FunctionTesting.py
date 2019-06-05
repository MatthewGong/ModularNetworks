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
    #BY = "species"
    #URL = 'sqlite:////home/matt/Documents/INL/ChemistryDiffraction/ChemistryDiffractionMasterBinned.db'
    #BATCH = 100

    data_query = EF.query_dataset(URL,BY,REL)

    ### Construct the dataset
    d_stack,c_stack,l_stack = EF.build_dataset(data_query,'d',BY)

    weights, num_classes= EF.make_weights(l_stack)

    folds_indicies = np.array_split(np.random.permutation(len(l_stack)),FOLDS)

    Diff_Aug = Augmentation.ShiftNoise()
    Chem_Aug = Augmentation.PercentNoise()
    
    #Augmentation
    #all_data = Augmentation.ChemDataset(d_stack,c_stack,c_stack,l_stack,d_aug=[Diff_Aug],c_aug=[Chem_Aug])
    #all_data = Augmentation.ChemDataset(d_stack,c_stack,c_stack,l_stack,d_aug=[Diff_Aug])
    #all_data = Augmentation.ChemDataset(d_stack,c_stack,c_stack,l_stack,c_aug=[Chem_Aug])
    
    #NoAugmentation
    all_data = Augmentation.ChemDataset(d_stack,c_stack,c_stack,l_stack)

    ### END DATASET CONSTRUCTION
     
    #model variants
    for k in range(FOLDS):
        mask = [m for m in range(FOLDS) if m!= k]
        
        test_data = Data.Subset(all_data,folds_indicies[k])

        chunked =np.concatenate([folds_indicies[f] for f in mask])
        
        train_data = Data.Subset(all_data,chunked)
        
        # fold_sampler makes it so it's already shuffled
        test_loaded = Data.DataLoader(dataset=test_data,batch_size=BATCH,shuffle=True)
        train_loaded = Data.DataLoader(dataset=train_data,batch_size=BATCH,shuffle=True)

        print(len(test_loaded.dataset),len(train_loaded.dataset))
        for i in range(1):
            ### TRAINING PARAMS
            #BATCH = 100
            #LR=.01
            num_layers += i
            EPOCHS += 1*i
            ### END TRAINING PARAMS

            start={"out":30+k*2,"stride":1,"k":3,"p":2,"length":900,"fold":k,"experiment":"all"}
            #start={"out":30+k*2,"stride":1,"k":3,"p":2,"length":540,"fold":k,"experiment":"all"}
            #name = str("{}foldsmallbatch{}".format(BY,k)) + "_d_"+ str(num_layers) 
            temp_name = name + str(k)
            
            # make the network
            diff_params = Factory.diff_param_maker(num_layers,mode="same",scale=1.3,start=start)
            ##bin_params = Factory.chem_param_maker(4,mode="triangle")
            per_params = Factory.chem_param_maker(4,mode="triangle")
            
            #####
            #Experiments
            #####

            #ExtraModular
            #after_params = Factory.after_param_maker(num_classes,3,[diff_params,diff_params],[bin_params,per_params],scale=2,mode="triangle")
            
            #DiffPer DiffPerNoAug
            after_params = Factory.after_param_maker(num_classes,3,[diff_params],[per_params],scale=2,mode="triangle")
            
            #DiffOnly
            #after_params = Factory.after_param_maker(num_classes,3,[diff_params],scale=2,mode="triangle")\

            #ChemOnly
            #after_params = Factory.after_param_maker(num_classes,3,[],[per_params],scale=2,mode="triangle")

            full_net = Nets.HybridNet([diff_params],[per_params],after_params)#DiffPer DiffPerNoAug
            #full_net = Nets.HybridNet([diff_params],[],after_params)#DiffOnly
            #full_net = Nets.HybridNet([],[per_params],after_params)#ChemOnly
            
            full_net.cuda()
            torchsummary.summary(full_net,input_size=[[(1,900)],[(1,118)]])
            #torchsummary.summary(full_net,input_size=[[(1,1)],[(1,118)]])
            
            loss_func = nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda())
            optimizer = torch.optim.SGD(full_net.parameters(), lr=LR, momentum=0.9)

            # run through data in splits
            for j in range(EPOCHS):
                running_loss = 0.0
                counter = 0
                for step,(diff,chem_bin,chem_per,label) in enumerate(train_loaded):
                    #print(diff.shape,chem_bin.shape)
                    output = full_net([diff.cuda().view(-1,1,900).float()],
                                    [chem_per.cuda().float()])

                    loss = loss_func(output.float(),label.view(-1).long().cuda())
                    
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
                        t_diff,t_chem_bin,t_chem_per,t_label = next(iter(test_loaded))
                        t_conv  = [t_diff.cuda().view(-1,1,900).float()]   
                        t_dense = [t_chem_per.cuda().float()]
                        EF.accuracy_detail(full_net,t_conv,t_dense,t_label)

            # Save the net for future evaluation
            torch.save(full_net,temp_name+ ".pt")
            t_diff,t_chem_bin,t_chem_per,t_label = next(iter(test_loaded))
            t_conv  = [t_diff.cuda().view(-1,1,900).float()]   
            t_dense = [t_chem_per.cuda().float()]
            EF.accuracy_detail(full_net,t_conv,t_dense,t_label)

            full_net.eval()
            a_list = []
            p_list = []

            for step,(diff,chem_bin,chem_per,label) in enumerate(test_loaded):
                #print x[0].shape,x[1].shape,x[2].shape,x[3].view(-1).shape
                #print step 
                #output = basic_net(diff.cuda().view(-1,1,900).float(),chem_per.cuda().float())
                output = full_net([diff.cuda().view(-1,1,900).float()],
                    [chem_per.cuda().float()])

                running_loss = 0.0
                actual, predicted = EF.accuracy_detail(full_net,
                                [diff.cuda().view(-1,1,900).float()],
                                [chem_per.cuda().float()],
                                label)
                
                a_list.append(actual)
                p_list.append(predicted)
            
            fig_size = (num_classes*2,num_classes*1.5)
            plt.figure(figsize=fig_size)
            conf_mat = confusion_matrix(np.vstack(a_list),np.vstack(p_list))

            divisor = conf_mat.astype(np.float).sum(axis=1)
            divisor[divisor==0] = 1
            conf_mat = conf_mat.T / divisor 
            print(conf_mat)
            np.savetxt("raw_mats/"+temp_name+".csv",conf_mat,delimiter=",")
            
            df_cm = pd.DataFrame(conf_mat, index = range(num_classes),columns = range(num_classes))
            
            font=20
            fig = sn.heatmap(df_cm, annot=True, fmt='.2f')
            plt.title(temp_name+'\n', fontsize=font)
            plt.xlabel("Expected",fontsize=4*font/5)
            plt.ylabel("Predicted", fontsize=4*font/5)
            #plt.show(fig)
            save_name = os.path.join('Images',temp_name) + '.png'
            plt.savefig(save_name)
            
            