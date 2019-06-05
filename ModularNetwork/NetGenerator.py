import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

import numpy as np

ELEMENTS_COUNT = 118
BIN_COUNT = 540



class Flatten2D(nn.Module):
    def forward(self,x):
        x = x.view(x.size()[0],1,-1)
        return x
class Flatten1D(nn.Module):
    def forward(self,x):
        x = x.view(x.size()[0],-1)
        return x

class BNConv1D(nn.Module):
    def __init__(self,in_f,out_f,p_size=2,k_size=3,stride=1):
        super(BNConv1D,self).__init__()

        self.conv = nn.Conv1d(in_f,out_f,k_size,stride,padding=k_size/2)
        self.pool = nn.MaxPool1d(p_size)
        self.bn = nn.BatchNorm1d(out_f)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.conv(x))
        x = self.pool(x)

        return self.bn(x)


class BNLinear(nn.Module):
    def __init__(self,in_f,out_f):
        super(BNLinear,self).__init__()
        self.linear = nn.Linear(in_f,out_f)
        self.bn = nn.BatchNorm1d(self.linear.out_features)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.linear(x))
        return self.bn(x)
"""
class BConv1D(nn.Module):
    def __init__(self,in_f,out_f,p_size=2,k_size=3,stride=1):
        super(BConv1D,self).__init__()

        self.conv = nn.Conv1d(in_f,out_f,k_size,stride,padding=k_size/2)
        self.pool = nn.MaxPool1d(p_size)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.conv(x))
        x = self.pool(x)

        return x
class BLinear(nn.Module):
    def __init__(self,in_f,out_f):
        super(BLinear,self).__init__()
        self.linear = nn.Linear(in_f,out_f)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.linear(x))
        return x
"""

class DenseNet(nn.Module):
    def __init__(self,ps):

        #PS is the list of paramters of layer sizes
        super(DenseNet, self).__init__()

        self.linearlist = nn.ModuleList([BNLinear(in_f=ps[i],out_f=ps[i+1]) for i in range(len(ps)-1)])  
       
    def forward(self,x):
        x = x.view(x.size()[0],-1)
        for i, l in enumerate(self.linearlist):
            x = l(x)
      
        return x

class ConvNet(nn.Module):
    def __init__(self,ps):
        super(ConvNet, self).__init__()
        self.convlist = nn.ModuleList([BNConv1D(in_f=ps[i]["in"],out_f=ps[i]["out"],
            p_size=ps[i]["p"],k_size=ps[i]["k"],stride=ps[i]["stride"]) for i in range(len(ps))]) 
        
        self.flat = Flatten1D()
        
    def forward(self,x):

        for i, l in enumerate(self.convlist):
            
            x = l(x)
        
        x = self.flat(x)
        return x

class Classification(nn.Module):

    def __init__(self,ps_after):
        super(Classification,self).__init__()
        self.classes = nn.ModuleList([BNLinear(ps_after[i],ps_after[i+1]) for i in range(len(ps_after)-1)])  

    def forward(self,z):
        
        for i, l in enumerate(self.classes):
            z = l(z)

        return z
        

class HybridNet(nn.Module):


    def __init__(self,Conv_Params=[],Dense_Params=[],Task_params=None):
        super(HybridNet,self).__init__()

        self.ConvModules = nn.ModuleList([ConvNet(Conv_Params[i]) for i in range(len(Conv_Params))])
        self.DenseModules = nn.ModuleList([DenseNet(Dense_Params[i]) for i in range(len(Dense_Params))])

        self.classes = Classification(Task_params)  

    def forward(self,conv,dense):
        
        # define the network for diffraction
        conv_out = []
        for i, module in enumerate(self.ConvModules):
            conv_out.append(module(conv[i]))

        dense_out = []
        for i, module in enumerate(self.DenseModules):
            dense_out.append(module(dense[i]))    

        # concatenate the last layers of the modules
        z = torch.cat(dense_out+conv_out,1)
        # run through the classification layers
        z = self.classes(z)

        return z

        