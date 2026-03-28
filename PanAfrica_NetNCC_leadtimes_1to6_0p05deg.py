import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
from datetime import date, datetime, timedelta
import xarray as xr
import netCDF4 as nc
from utils.u_interpolate_small import regrid_irregular_quick
from utils import u_interpolate_small as uint
import glob
import calendar
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import math
import time
from utils.netncc_functions import FSS_accuracy_metric_gpu, FSS_loss_gpu, create_mean_filter, select_model_order, plot_maps_colorbar
from utils.plot_functions import plot_maps_colorbar, plot_maps_colorbar_lsta
from utils.netncc_classes import ConvLayer, netncc, XarrayUNetDataset, evalUNetDataset
from scipy.ndimage import uniform_filter


##
# Define domain and time period
# date is supposed to change automatically
current_date = '202601191400'
#--------------------------------
current_year = current_date[0:4]
current_month = current_date[4:6]
current_day = current_date[6:8]


###
domains =['WA','SA','EA']  # IMPORTANT - donot change the order as it affects the merging of domains into one image
device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TIR data dir
dir_name = '/prj/nflics/real_time_data/'+current_year+'/'

###### Define input shape
image_height= 1024 #lat
image_width= image_height #lon
in_channels = 3
out_channels = 1
start_year = '2004'
end_year = '2023'
leadtimes= [1,2,3,4,5,6]

# grid
resolution = 0.05
start_lat = -36.50 #
end_lat = 27.95  #
start_lon = -20.05 #
end_lon = 50.75 #

reg_lat_PA = np.round(np.arange(start_lat, end_lat, resolution),2)
reg_lon_PA = np.round(np.arange(start_lon, end_lon, resolution),2) 

# select files for input
current_date_int = datetime.strptime(current_date, '%Y%m%d%H%M')
to_date=datetime.strptime(str(current_date), '%Y%m%d%H%M')
to_minus_1hr_date=current_date_int-timedelta(hours=1)
to_minus_1hr_date= to_minus_1hr_date.strftime('%Y%m%d%H%M')
to_minus_2hr_date=current_date_int-timedelta(hours=2)
to_minus_2hr_date= to_minus_2hr_date.strftime('%Y%m%d%H%M')

dates_of_interest = [to_minus_2hr_date,to_minus_1hr_date,str(current_date)]
    

# list of files to be read
list_of_files = []                
for l in range(0,len(dates_of_interest),1):
    dates_of_interest_curr = dates_of_interest[l]
    list_of_files.append(dir_name+dates_of_interest_curr[4:6]+'/'+dates_of_interest_curr[6:8]+'/IR_108_BT_'+dates_of_interest_curr[0:4]+dates_of_interest_curr[4:6]+dates_of_interest_curr[6:8]+'_'+dates_of_interest_curr[8:]+'_eumdat.nc')

# check for to-2 file
if os.path.exists(list_of_files[0]) == False:
    to2_date = dates_of_interest[0]
    dir_name = '/prj/nflics/real_time_data/'+current_year+'/'+to2_date[4:6]+'/'+to2_date[6:8]+'/' 
    all_file_names = sorted(glob.glob(dir_name+"IR*.nc"));  #
    latest_to2_file = all_file_names[-4*2] 
   
    # check time between files 
    to_2_date=latest_to2_file[-23:-15]+latest_to2_file[-14:-10]
    to_2_datetime=datetime.strptime(str(int(to_2_date)), '%Y%m%d%H%M')
    time_difference = to_date-to_2_datetime    
    if time_difference< timedelta(hours=4.1): ###### IMPORTANT#####
        list_of_files[0]=latest_to2_file
        list_of_files[1]=all_file_names[-4]
    else:
        list_of_files[0]=list_of_files[2]
        list_of_files[1]=list_of_files[2]   
    
def predict_for_region(domain, to_date, dates_of_interest, list_of_files,leadtimes):
    # hardcoded for now
    image_height= 1024 #lat
    image_width= image_height #lon
    in_channels = 3
    out_channels = 1
    start_year = '2004'
    end_year = '2023'
    t = 3 #prior hours for prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    ###
    if domain=='WA':
        moi = 'JAS'
        end_lat = 27.95  #
        start_lon = -20.05 #
    
    elif domain=='SA':
        moi = 'DJF'
        end_lat = 14.7  #
        start_lon = -0.4 #
    
    elif domain== 'EA':
        moi = 'MAM'
        end_lat = 27.95  #
        start_lon = -0.4 #
  
    # load indices for domain and corresponding PA indices
    with open('utils/'+domain+'_regridding_weights_TIR_0p05.pkl', 'rb') as file:
            inds, weights, shape = pickle.load(file) 
    with open('utils/'+domain+'_regridding_weights_TIR_0p05_PA.pkl', 'rb') as file:
            inds_pa, weights_pa, shape_pa= pickle.load(file) 
    
    # read input for prediction
    regridded_tir = np.zeros((t,image_height,image_height),dtype=float) 
    
    # read in tir data
    for l in range(0,len(list_of_files),1): 
        tir_filename = list_of_files[l]
        if os.path.exists(tir_filename):
            ds = xr.open_dataset(tir_filename).squeeze() 
            tir_temp =  ds['ir108_bt'].values  #
            regridded_tir[l,:,:] = uint.interpolate_data(tir_temp, inds, weights, shape)  # interpolation using saved weights for MSG TIR
           
    # input data
    ind_tir = np.where(regridded_tir>-0.01)
    regridded_tir[ind_tir] = 0
    regridded_tir[np.isnan(regridded_tir)] = 0
    regridded_tir[regridded_tir<-110] = 0 #-110
    x_pred = np.round(regridded_tir/-110,4)
    test_image= torch.tensor(x_pred.astype(np.float32))
    test_image = test_image.unsqueeze(0)   # Add
    
    
    # predict for all leadtimes
    filtered_image=[]
    actual_cores=[]
    
    for leadtime in leadtimes:
        # read in model
        modelFile= 'trained_models/'+domain+'/'+domain+'_'+moi+'_2004_to_2023_'+str(leadtime)+'hr_using_1hr_0p05deg.pth'
        unet_model = netncc()# Training parameters
        unet_model = nn.DataParallel(unet_model)
        unet_model = unet_model.to(device)
        unet_model.load_state_dict(torch.load(modelFile,map_location=device))
        # Set the model to evaluation mode
        unet_model.eval()
        
        prediction_time = int((to_date+timedelta(hours=leadtime)).strftime('%Y%m%d%H%M'))  ###### IMPORTANT#####
        # time of day predicted
        time_of_day_tr= np.zeros((1,image_height, image_width))
        time_of_day = float(str(prediction_time)[8:])/2345
        time_of_day_tr[:,:,:]=round(np.sin(time_of_day*math.pi),2)
        tod = torch.tensor(time_of_day_tr.astype(np.float32))
        tod = tod.unsqueeze(0)   # Add batch dimension
    
        # predict for one day
        with torch.no_grad():
            predicted_core = unet_model(test_image,tod)
        
        predicted_core = torch.nan_to_num(predicted_core, nan=0.0, posinf=0, neginf=0).squeeze()
        filtered_image.append(torch.Tensor.numpy(predicted_core)) # in case filtering needed later on

    filtered_image = np.array(filtered_image)
    
    regrid_nowcasts = uint.interpolate_data(filtered_image, inds_pa, weights_pa, shape_pa)  # interpolation using saved weights for MSG TIR
    
    return regrid_nowcasts 

# predict for all domains
all_domain_nowcasts = np.zeros((len(domains), len(leadtimes), len(reg_lat_PA), len(reg_lon_PA)))
panAfrica_nowcast =  np.zeros((len(leadtimes), len(reg_lat_PA), len(reg_lon_PA)))

for i in range(len(domains)):
    all_domain_nowcasts[i,:,:,:] = predict_for_region(domains[i],to_date,dates_of_interest,list_of_files,leadtimes)

# merge into one map for each leadtime    
domain_order =  select_model_order(int(current_month))
for d in [2,1,0]:
    domain_image = all_domain_nowcasts[domain_order[d],:,:,:].squeeze()
    ind_ok = ~np.isnan(domain_image)
    panAfrica_nowcast[ind_ok] = domain_image[ind_ok]
    del ind_ok
    
panAfrica_nowcast[np.isnan(panAfrica_nowcast)]=0
smoothed_nowcasts = uniform_filter(panAfrica_nowcast, size=3, mode='reflect')

# smoothed_nowcasts needs to be convected into a geotiff and shown on the portal
    