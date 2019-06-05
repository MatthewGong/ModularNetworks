import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data.sampler import Sampler,SubsetRandomSampler
import torchvision
from torchsummary import summary

import os
import sys

import seaborn as sn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast
import numpy as np
import pandas as pd
from sqlalchemy import Column, ForeignKey, Integer, String, Table
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from binned_schema import DiffChemBinned, Base

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import customtorchsummary as torchsummary
import ModularNetGenerator as Nets
import DatasetAugmentation as DA
import NetManufacturing as Factory


def accuracy_detail(model, diff, c, data_y):
    oupt = model(diff,c)
    (big_val, big_idx) = torch.max(oupt, dim=1)
    predicted = big_idx.view(big_idx.shape[0],1).cpu().numpy().copy()
    actual = data_y.cpu().numpy().copy()
    correct = big_idx.view(big_idx.shape[0],1).cuda()
    correct -= data_y.cuda()
    n_correct = torch.sum(correct==0).cpu().numpy()
    n_wrong = torch.sum(correct!=0).cpu().numpy()

    return predicted, actual

#bin_engine = create_engine('sqlite:////home/matt/Documents/INL/ChemistryDiffraction/ChemistryDiffractionMasterBinned2.db')
bin_engine = create_engine('sqlite:////home/matt/Documents/INL/ChemistryDiffraction/ChemistryDiffractionMasterBinned.db')

Bin_Session = sessionmaker()

Bin_Session.configure(bind=bin_engine)

bin_session = Bin_Session()

data_query = bin_session.query(DiffChemBinned.theta_binned,
                               DiffChemBinned.d_binned,
                               DiffChemBinned.chem_simple,
                               DiffChemBinned.chem_count,
                               DiffChemBinned.chem_binary,
                               DiffChemBinned.genera,
                               DiffChemBinned.fam_num)

### TRAINING PARAMS
EPOCHS= 5
BATCH = 500
LR=.01
ELEMENTS_COUNT = 118
## END TRAINING PARAMS
print(data_query.count())

#data_query = data_query.filter(DiffChemBinned.fam_num!=1)
#data_query = data_query.filter(DiffChemBinned.fam_num!=2)

print(data_query.count())

counter = 0
d_stack = []
c_stack = []
l_stack = []


### Construct the dataset
for entry in data_query.yield_per(1000):
    #d_bin = np.fromstring(entry.d_binned[1:-1],sep=',')
    theta_bin = np.fromstring(entry.theta_binned[1:-1],sep=',')

    
    counter += 1
    #if sum(d_bin) > 0:
    if sum(theta_bin) > 0:
        arr_sum = np.fromstring(entry.chem_count[1:-1],sep=",")

        c_stack.append(arr_sum/np.sum(arr_sum))    

        #d_stack.append(d_bin)
        d_stack.append(theta_bin)
        fam = entry.fam_num
        l_stack.append([fam])

        if (counter+1)%10000 ==0:
            print counter

d_data = np.vstack(d_stack)
c_data = np.vstack(c_stack)
l_data = np.vstack(l_stack)
l_data -= 1 # necessary for family since indexing starts at 1

	

    ### Making the weights
print np.unique(l_data,return_counts=True)
selected, counts = np.unique(l_data,return_counts=True) 
total = np.sum(counts)
weights = []
for i in range(len(counts)):
    weights.append(float(total)/counts[i])
print weights
    ### 



# Create SubsetRandomSampler instead of splitting data, save splits for futre comparison

#fold_sampler = SubsetRandomSampler(np.concatenate(folds_indicies[1:]))
#fold_sampler_test = SubsetRandomSampler(np.concatenate(folds_indicies[0]))
folds_indicies = np.array_split(np.random.permutation(int(total)),10)

Shift_Aug = DA.ShiftNoise()
Drop_Aug = DA.DropNoise()
#all_data = DA.ChemDataset(d_data,c_data,c_data,l_data,d_aug=[Drop_Aug,Shift_Aug])
all_data = DA.ChemDataset(d_data,c_data,c_data,l_data)


train_data = Data.Subset(all_data,np.concatenate(folds_indicies[1:]))
test_data = Data.Subset(all_data,folds_indicies[0])

#all_data = Noneself.diffraction_augment
#d_stack, c_stack, l_stack = None,None,None # free up memory

# fold_sampler makes it so it's already shuffled
test_loaded = Data.DataLoader(dataset=test_data,batch_size=BATCH,shuffle=True)
train_loaded = Data.DataLoader(dataset=train_data,batch_size=BATCH,shuffle=True)
print(len(train_loaded.dataset),len(test_loaded.dataset))
#test_loaded = Data.DataLoader(dataset=all_data,batch_size=BATCH,sampler=fold_sampler_test)

### END DATASET CONSTRUCTION


#model variants
for k in range(10):
    for i in range(6):
        num_classes = 7
        num_layers = 6 + i
        start={"out":30+k*2,"stride":1,"k":3,"p":2,"length":900}
        name = str(start) + "_theta_"+ str(num_layers) + ".pt"
        diff_params = Factory.diff_param_maker(num_layers,mode="triangle",scale=1.3,start=start)
        bin_params = Factory.chem_param_maker(4,mode="triangle")
        per_params = Factory.chem_param_maker(4,mode="triangle")
        after_params = Factory.after_param_maker(num_classes,5,[diff_params,diff_params,diff_params],
            [bin_params,per_params],scale=3,mode="triangle")
        full_net = Nets.HybridNet([diff_params,diff_params,diff_params],[bin_params,per_params],after_params)
        full_net.cuda()
        print(diff_params)
        loss_func = nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda())
        optimizer = torch.optim.SGD(full_net.parameters(), lr=LR, momentum=0.9)
        #torchsummary.summary(full_net,input_size=[[(1,900),(1,900),(1,900)],[(1,118),(1,118)]])
        
        #print os.system("nvidia-smi")
        # run through data in splits
        for j in range(EPOCHS):
            running_loss = 0.0
            counter = 0
            for step,(diff,chem_bin,chem_per,label) in enumerate(train_loaded):
                #print x[0].shape,x[1].shape,x[2].shape,x[3].view(-1).shape
                #print step
                #output = basic_net(diff.cuda().view(-1,1,900).float(),chem_per.cuda().float())
                x = [diff.cuda().view(-1,1,900).float(),diff.cuda().view(-1,1,900).float(),diff.cuda().view(-1,1,900).float()]
                y =  [chem_bin.cuda().float(),chem_per.cuda().float()]
                output = full_net(x,y)
                #print output
                #basic_net.zero_grad()
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
                    t_x =[t_diff.cuda().view(-1,1,900).float(),t_diff.cuda().view(-1,1,900).float(),t_diff.cuda().view(-1,1,900).float()]
                    t_y =[t_chem_bin.cuda().float(),t_chem_per.cuda().float(),t_chem_bin.cuda().float(),t_chem_per.cuda().float()]
                    accuracy_detail(full_net, t_x, t_y, t_label)

        # Save the net for future evaluation
        torch.save(full_net,name)
        #test_net = torch.load(name)
        t_diff,t_chem_bin,t_chem_per,t_label = next(iter(test_loaded))
        accuracy_detail(full_net,
                        [t_diff.cuda().view(-1,1,900).float(),t_diff.cuda().view(-1,1,900).float(),t_diff.cuda().view(-1,1,900).float()],
                        [t_chem_bin.cuda().float(),t_chem_per.cuda().float()],
                        t_label)
        full_net.eval()
        a_list = []
        p_list = []
        for step,(diff,chem_bin,chem_per,label) in enumerate(test_loaded):
            #print x[0].shape,x[1].shape,x[2].shape,x[3].view(-1).shape
            #print step
            #output = basic_net(diff.cuda().view(-1,1,900).float(),chem_per.cuda().float())
            #output = full_net([diff.cuda().view(-1,1,900).float(),diff.cuda().view(-1,1,900).float(),diff.cuda().view(-1,1,900).float()],
            #    [chem_bin.cuda().float(),chem_per.cuda().float()])

            running_loss = 0.0
            actual, predicted = accuracy_detail(full_net,
                            [diff.cuda().view(-1,1,900).float(),diff.cuda().view(-1,1,900).float(),diff.cuda().view(-1,1,900).float()],
                            [chem_bin.cuda().float(),chem_per.cuda().float()],
                            label)
            
            a_list.append(actual)
            p_list.append(predicted)
        
        fig_size = (num_classes*2,num_classes*1.5)
        plt.figure(figsize=fig_size)
        conf_mat = confusion_matrix(np.vstack(a_list),np.vstack(p_list))

        divisor = conf_mat.astype(np.float).sum(axis=1)
        divisor[divisor==0] = 1
        conf_mat = conf_mat.T / divisor 
        df_cm = pd.DataFrame(conf_mat, index = range(num_classes),columns = range(num_classes))
        
        font=20
        fig = sn.heatmap(df_cm, annot=True, fmt='.2f')
        plt.title(name+'\n', fontsize=font)
        plt.xlabel("Expected",fontsize=4*font/5)
        plt.ylabel("Predicted", fontsize=4*font/5)
        plt.show(fig,block=False)
        save_name = os.path.join('Images',name) + '.png'
        plt.savefig(save_name)