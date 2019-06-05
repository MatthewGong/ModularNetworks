from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from binned_schema import DiffChemBinned, Base
from AssociatedSchema import DiffEntry,Base
import numpy as np
import torch

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

def query_dataset(path='sqlite:////home/gongm/DiffChem.db',mode="family",selection=6,ignore=[]):
    print("establishing connection to database")
    engine = create_engine(path)

    print("1/3")
    Bin_Session = sessionmaker()
    print("2/3")
    Bin_Session.configure(bind=engine)
    print("3/3")
    bin_session = Bin_Session()

    print("established connection to database")
    print("querying database")
    
    data_query = bin_session.query(DiffChemBinned.theta_binned,
                                   DiffChemBinned.d_binned,
                                   DiffChemBinned.chem_simple,
                                   DiffChemBinned.chem_count,
                                   DiffChemBinned.chem_binary,
                                   DiffChemBinned.genera,
                                   DiffChemBinned.fam_num,
                                   DiffChemBinned.gen_rel,
                                   DiffChemBinned.sgs_rel,
                                   DiffChemBinned.sgs_num
                                   )
    """
    data_query = bin_session.query(DiffEntry.theta_binned,
                                   DiffEntry.d_binned,
                                   DiffEntry.chem_simple,
                                   DiffEntry.genera,
                                   DiffEntry.fam_num,
                                   DiffEntry.gen_rel,
                                   DiffEntry.sgs_rel)
    """
    print("queried database")
    print(data_query.count())

    print("filtering data")
    if mode=="family":
        pass # this is the default
    elif mode=="genera":
        data_query = data_query.filter(DiffChemBinned.fam_num==selection)
    elif mode=="species":
        data_query = data_query.filter(DiffChemBinned.genera==selection)
    else:
        raise ValueError("wrong mode")
    


    for i in ignore:
        if mode=="family":
           data_query = data_query.filter(DiffChemBinned.fam_num!=i)
        elif mode=="genera":
            data_query = data_query.filter(DiffChemBinned.gen_rel!=i)
        elif mode=="species":
            data_query = data_query.filter(DiffChemBinned.sgs_rel!=i)
        else:
            raise ValueError("wrong mode")
    

    print("filtered data")
    print(data_query.count())

    return data_query

def build_dataset(data_query,mode,heirarchy):
    """
    DatasetBuilding
    """
    counter = 0
    b_stack = []
    c_stack = []
    l_stack = []

    print("Building Dataset")
    
    for entry in data_query.yield_per(1000):

        if mode=="d":
            binned_data = np.fromstring(entry.d_binned[1:-1],sep=',')
        elif mode=="theta":
            binned_data = np.fromstring(entry.theta_binned[1:-1],sep=',')
        else:
            raise ValueError("{} not a recognized mode. use 'd' or 'theta'".format(mode))
        #print(binned_data.shape)
        counter += 1

        if sum(binned_data) > 0:
            #if sum(theta_bin) > 0:
            arr_sum = np.fromstring(entry.chem_count[1:-1],sep=",")

            c_stack.append(arr_sum)#/np.sum(arr_sum))    

            b_stack.append(binned_data)
            #d_stack.append(theta_bin)
            if heirarchy == "family":
                shift = 1
                label = entry.fam_num
            elif heirarchy == "genera":
                shift = 0
                label = entry.gen_rel
            elif heirarchy == "species":
                shift = 0
                label = entry.sgs_rel
            else:
                raise ValueError("Wrong heirarchy")


            l_stack.append([label])

            if (counter+1)%10000 ==0:
                print(counter)


    l_data = np.vstack(l_stack)
    l_data -= shift # necessary for family since indexing starts at 1

    return  np.vstack(b_stack),np.vstack(c_stack),l_data

def make_weights(l_data):
    ### Making the weights based on prescence in dataset
    print(np.unique(l_data,return_counts=True))
    selected, counts = np.unique(l_data,return_counts=True) 
    total = np.sum(counts)
    print(total,len(l_data))
    weights = []
    for i in range(len(counts)):
        weights.append(float(total)/counts[i])
    print(weights)
    num_classes = len(weights)
    return weights,num_classes
    ### endWights


