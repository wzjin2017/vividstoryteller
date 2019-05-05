import h5py
import os
import numpy as np
f1 = h5py.File('/home/weizhaojin/Downloads/flowers.hdf5','r+')
#Get the HDF5 group
group = f1['train']

#Checkout what keys are inside that group.
#for key in group.keys():
#    print(key)
data = group['image_00001_3']
for key in data.keys():
    print(key)
    print(data[key])


#Do whatever you want with data

#After you are done
f1.close()