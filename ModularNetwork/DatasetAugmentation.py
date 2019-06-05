import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

from torchvision.transforms import Compose
import numpy as np

ELEMENTS_COUNT = 118
BIN_COUNT = 540

class ChemDataset(Data.Dataset):
    'characterizes a dataset for use by Pytorch'
    def __init__(self, Diff, ChemB,ChemP,Labels, d_aug=None,c_aug=None):
        
        'Initialize the variables'
        self.Diffration = Diff
        self.Chem_bin = ChemB
        self.Chem_per = ChemP

        self.Labels = Labels

        if d_aug:
            self.diffraction_augment = Compose(d_aug)
        else:
            self.diffraction_augment = d_aug
        if c_aug:
            self.chemistry_augment = Compose(c_aug)
        else:     
            self.chemistry_augment = c_aug

    def __len__(self):

        return len(self.Labels)        

    def __getitem__(self, index):
        'Explicitily defines how to return an item'
        
        diff = self.Diffration[index]
        chem_bin = self.Chem_bin[index]
        chem_per = self.Chem_per[index]

        label = self.Labels[index]

        if self.diffraction_augment:
            #for t in self.diffraction_augment: 
            diff = self.diffraction_augment(diff)

        if self.chemistry_augment:
            #chem_bin = self.chemistry_augment(chem_bin)
            chem_per = self.chemistry_augment(chem_per)
        
        return diff,chem_bin,chem_per,label

class ShiftNoise():
    def __init__(self):
        # +- 4 bins
        # normally distributed 
        pass

    def __call__(self, sample):
        shift = int(np.round(4*np.random.normal(scale=.3)))

        if shift == 0:
            return sample     
        elif shift < 0:
            sample = np.pad(sample,pad_width=[0,abs(shift)],mode='constant',constant_values=0)
            sample =  sample[abs(shift):]
        else:
            sample = np.pad(sample,pad_width=[shift,0],mode='constant',constant_values=0)
            sample = sample[:-(shift)]

        return sample

class DropNoise():
    def __init__(self):
        # drop out peaks by prominence  
        # 
        pass

    def __call__(self, sample):


        return sample    

class PercentNoise():
    def __init__(self):
        pass

    def __call__(self, sample):
        perc_mask = sample > 0
        
        sample[perc_mask] += np.random.normal(0,.02,len(sample[perc_mask])) 
        
        sample /= np.sum(sample)
        
        return sample 
