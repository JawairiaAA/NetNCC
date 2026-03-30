# classes for netncc evaluation- TIR only 512x512 West africa domain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, datetime, timedelta
import xarray as xr
import netCDF4 as nc
#from u_interpolate_small import regrid_irregular_quick
from utils.u_interpolate_small import regrid_irregular_quick
from utils import u_interpolate_small as uint
#from ndays import numOfDays
import pickle
import glob
import calendar
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import math
import time
from utils.netncc_functions import FSS_accuracy_metric_gpu, FSS_loss_gpu, create_mean_filter

#### temp
# Define domain and time period
start_year = '2004'
end_year = '2023'
#start_month = '07' #
#end_month = '09' #start_month # '06'  #
#start_day = '01'
#end_day = '31'

image_height= 1024 #lat
image_width= image_height #lon
in_channels = 3
out_channels = 1


#inds, weights, shape = uint.interpolation_weights(mlon, mlat, reg_lon, reg_lat) # save weights for continuous use - MSG interpolation on regular. 
##########



# Define conv layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
        
# Define the U-Net model
class netncc(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(netncc, self).__init__()
        self.down2 = ConvLayer(in_channels, 4)
        self.down2_pool = nn.MaxPool2d(2)
        self.down3 = ConvLayer(4, 8)
        self.down3_pool = nn.MaxPool2d(4)
        self.center = ConvLayer(8, 16)
        self.center2 = ConvLayer(16, 8)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3_conv = ConvLayer(16, 4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2_conv = ConvLayer(9, 4)   #9
        self.output = nn.Conv2d(4, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, meta_inputs):
        down2 = self.down2(inputs)
        down2_pool = self.down2_pool(down2)
        down3 = self.down3(down2_pool)
        down3_pool = self.down3_pool(down3)
        center = self.center(down3_pool)
        center = self.center2(center)
        up3 = self.up3(center)
        up3 = torch.cat([down3, up3], dim=1)
        up3 = self.up3_conv(up3)
        up2 = self.up2(up3)
        up2 = torch.cat([down2, up2, meta_inputs], dim=1)
        up2 = self.up2_conv(up2)
        output = self.output(up2)
        output = self.sigmoid(output)
        return output


# Dataset class using xarray
class XarrayUNetDataset(Dataset):
    def __init__(self, input_files, output_files,domain):
        self.image_files = input_files #
        self.mask_files = output_files #
        self.domain = domain #
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        if idx<2*4:
            idx=2*4
        image_paths = [self.image_files[idx-2*4],self.image_files[idx-1*4],self.image_files[idx]] # to
        mask_path = self.mask_files[idx] # to
        time_tir = image_paths[2][-15:-3]
        #print(image_paths)

        if self.domain== 'WA':
            with open('utils/WA_regridding_weights_TIR_0p05.pkl', 'rb') as file:
                inds, weights, shape = pickle.load(file) 
        elif self.domain == 'SA':
            with open('utils/SA_regridding_weights_TIR_0p05.pkl', 'rb') as file:
                inds, weights, shape = pickle.load(file) 
        else:
            with open('utils/EA_regridding_weights_TIR_0p05.pkl', 'rb') as file:
                inds, weights, shape = pickle.load(file) 

        
        # pre allocate arrays
        regridded_tir = np.zeros((3,image_height,image_width),dtype=float) 
        x_train = np.zeros((in_channels,image_height,image_width),dtype=float) 

        # select TIR three files (selecting past files from list now) [channels, height, width]
        for i in range(3):
            image_ds = xr.open_dataset(image_paths[i]).squeeze() 
            tir_temp =  image_ds['tir'].values  #
            regridded_tir[i,:,:]= uint.interpolate_data(tir_temp, inds, weights, shape)  # interpolation using saved 
            
        # input data
        regridded_tir[np.isnan(regridded_tir)] = 0
        ind_tir = np.where(regridded_tir>-0.01)
        regridded_tir[ind_tir] = 0
        regridded_tir[regridded_tir<-110] = 0 #-110
        x_train = np.round(regridded_tir/-110,4)
        
        # open and read files- cores output
        mask_ds = xr.open_dataset(mask_path).squeeze() 
        core_temp = mask_ds['cores'].values
        regridded_cores = uint.interpolate_data(core_temp, inds, weights, shape)  # interpolation using saved weights for MSG TIR
        ind = np.where(regridded_cores>0)
        regridded_cores[ind] = 1
        regridded_cores[np.isnan(regridded_cores)] = 0
        y_train=np.expand_dims(regridded_cores, axis=0)

        #time_core = str(mask_path[-15:-3])
        #prediction_time = time_core
        time_of_day_tr= np.zeros((1,image_height, image_width))
        time_of_day = float(str(mask_path[-15:-3])[8:])/2345
        time_of_day_tr[:,:,:]=round(np.sin(time_of_day*math.pi),4)

        return torch.tensor(x_train.astype(np.float32)), torch.tensor(time_of_day_tr.astype(np.float32)), torch.tensor(y_train.astype(np.float32))


# Dataset class using xarray
class evalUNetDataset(Dataset):
    def __init__(self, input_files, output_files,domain):
        self.image_files = input_files #
        self.mask_files = output_files #
        self.domain = domain #
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        if idx<2*4:
            idx=2*4
        image_paths = [self.image_files[idx-2*4],self.image_files[idx-1*4],self.image_files[idx]] # to
        mask_path = self.mask_files[idx] # to
        time_tir = image_paths[2][-15:-3]
        #print(image_paths)

        if self.domain== 'WA':
            with open('utils/WA_regridding_weights_TIR_0p05.pkl', 'rb') as file:
                inds, weights, shape = pickle.load(file) 
        elif self.domain == 'SA':
            with open('utils/SA_regridding_weights_TIR_0p05.pkl', 'rb') as file:
                inds, weights, shape = pickle.load(file) 
        else:
            with open('utils/EA_regridding_weights_TIR_0p05.pkl', 'rb') as file:
                inds, weights, shape = pickle.load(file) 

        
        # pre allocate arrays
        regridded_tir = np.zeros((3,image_height,image_width),dtype=float) 
        x_train = np.zeros((in_channels,image_height,image_width),dtype=float) 

        # select TIR three files (selecting past files from list now) [channels, height, width]
        for i in range(3):
            image_ds = xr.open_dataset(image_paths[i]).squeeze() 
            tir_temp =  image_ds['tir'].values  #
            regridded_tir[i,:,:]= uint.interpolate_data(tir_temp, inds, weights, shape)  # interpolation using saved weights for MSG TIR
 
        # input data
        ind_tir = np.where(regridded_tir>-0.01)
        regridded_tir[ind_tir] = 0
        regridded_tir[np.isnan(regridded_tir)] = 0
        regridded_tir[regridded_tir<-110] = 0 #-110
        x_train = np.round(regridded_tir/-110,4)
        tod_to = str(image_paths[i][-15:-3])
       
        # cores at to
        core_temp = image_ds['cores'].values
        regridded_cores_to = uint.interpolate_data(core_temp, inds, weights, shape)  # interpolation using saved weights for MSG TIR
        ind = np.where(regridded_cores_to>0)
        regridded_cores_to[ind] = 1
        regridded_cores_to[np.isnan(regridded_cores_to)] = 0
        #regridded_cores_to= regridded_cores_to[:a,b:]
        
        # open and read files- cores output
        mask_ds = xr.open_dataset(mask_path).squeeze() 
        core_temp = mask_ds['cores'].values
        regridded_cores = uint.interpolate_data(core_temp, inds, weights, shape)  # interpolation using saved weights for MSG TIR
        ind = np.where(regridded_cores>0)
        regridded_cores[ind] = 1
        regridded_cores[np.isnan(regridded_cores)] = 0
        y_train=np.expand_dims(regridded_cores, axis=0)

        #time_core = str(mask_path[-15:-3])
        #prediction_time = time_core
        time_of_day_tr= np.zeros((1,image_height, image_width))
        time_of_day = float(str(mask_path[-15:-3])[8:])/2345
        time_of_day_tr[:,:,:]=round(np.sin(time_of_day*math.pi),4)

        return torch.tensor(x_train.astype(np.float32)), torch.tensor(time_of_day_tr.astype(np.float32)), torch.tensor(y_train.astype(np.float32)), tod_to, regridded_cores_to



        