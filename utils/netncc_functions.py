import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
import pickle 
from datetime import date, datetime, timedelta
import xarray as xr
import netCDF4 as nc
#from u_interpolate_small import regrid_irregular_quick
#import u_interpolate_small as uint
import glob
import calendar
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import math
import time
## a clean way of plotting - use matplotlib functions directly:
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy

#############################
#if torch.cuda.is_available():
#    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
#else:
#    print("No GPU available. Training will run on CPU.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
############################

def draw_map(ax, data, lon, lat, title=None,  mask_sig=None, quiver=None, contour=None, cbar_label=None, **kwargs):
    mapp = ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), **kwargs)  # this is the actual plot
    ## mask for significance indicator
    if mask_sig is not None:
         plt.contourf(lon, lat, mask_sig, colors='none', hatches='.',
                     levels=[0.5, 1], linewidth=0.1)
    ## quiver list
    if quiver is not None:
        qu = ax.quiver(quiver['x'], quiver['y'], quiver['u'], quiver['v'], scale=quiver['scale'])
    ## additional contour on plot   
    if contour is not None:
        ax.contour(contour['x'], contour['y'], contour['data'], levels=contour['levels'], cmap=contour['cmap'] )
    ax.coastlines()   ## adds coastlines
    # Gridlines
    xl = ax.gridlines(draw_labels=True);   # adds latlon grid lines
    xl.top_labels = False   ## labels off
    xl.right_labels = False
    plt.title(title)
    # Countries
    ax.add_feature(cartopy.feature.BORDERS, linestyle='--'); # adds country borders
    cbar = plt.colorbar(mapp,shrink=0.6)  # adds colorbar
    cbar.set_label(cbar_label)

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

def plot_maps_colorbar(ax, lon, lat, data, title,levels_custom, plot_label,cmap):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.75, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    #gl.xlocator = mticker.FixedLocator([20, 25, 30, 35, 40])
    #gl.ylocator = mticker.FixedLocator([0, -5, -10, -15, -20, -25])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 7, 'color': 'gray'}
    gl.ylabel_style = {'size': 7, 'color': 'gray'}
    # ax.set_title(title)
    # Countries
    m=ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), levels=levels_custom,cmap=cmap)  # this is the actual plot
    ax.coastlines()   ## adds coastlines
    ax.add_feature(cartopy.feature.BORDERS, linestyle='--',color='white'); # adds country borders
    cbar = plt.colorbar(m,fraction=0.03)  # adds colorbar
    cbar.set_label(plot_label)
    
# Define FSS loss function
def create_mean_filter(half_num_rows, half_num_columns, num_channels):
    num_rows = 2 * half_num_rows + 1
    num_columns = 2 * half_num_columns + 1
    weight = 1. / (num_rows * num_columns)
    return torch.full((num_channels, num_channels, num_rows, num_columns), weight, dtype=torch.float32)

def FSS_loss_custom_training_filter(predicted_tensor, target_tensor, half_window_size_px, use_as_loss_function=True):
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    weight_matrix=weight_matrix.to(device)
    smoothed_target_tensor = F.conv2d(target_tensor, weight_matrix, padding=half_window_size_px)
    smoothed_prediction_tensor = F.conv2d(predicted_tensor, weight_matrix, padding=half_window_size_px)    
    actual_mse = torch.mean((smoothed_target_tensor - smoothed_prediction_tensor) ** 2)
    reference_mse = torch.mean(smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2)
    if use_as_loss_function:
        return actual_mse / reference_mse
    return 1. - actual_mse / reference_mse
    
def FSS_loss(predicted_tensor, target_tensor, half_window_size_px=2, use_as_loss_function=True):
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    smoothed_target_tensor = F.conv2d(target_tensor, weight_matrix, padding=half_window_size_px)
    smoothed_prediction_tensor = F.conv2d(predicted_tensor, weight_matrix, padding=half_window_size_px)    
    actual_mse = torch.mean((smoothed_target_tensor - smoothed_prediction_tensor) ** 2)
    reference_mse = torch.mean(smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2)
    if use_as_loss_function:
        return actual_mse / reference_mse
    return 1. - actual_mse / reference_mse
    
def FSS_loss_gpu(predicted_tensor, target_tensor, half_window_size_px=2, use_as_loss_function=True):
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    weight_matrix=weight_matrix.to(device)
    smoothed_target_tensor = F.conv2d(target_tensor, weight_matrix, padding=half_window_size_px)
    smoothed_prediction_tensor = F.conv2d(predicted_tensor, weight_matrix, padding=half_window_size_px)    
    actual_mse = torch.mean((smoothed_target_tensor - smoothed_prediction_tensor) ** 2)
    reference_mse = torch.mean(smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2)
    if use_as_loss_function:
        return actual_mse / reference_mse
    return 1. - actual_mse / reference_mse

def calc_FSS_pixelwise(forecast, observed):
    actual_mse = np.round(np.mean((forecast - observed) ** 2, axis=0),3)
    reference_mse = np.round(np.mean(forecast ** 2 + observed ** 2, axis=0),3)
    return 1 - (actual_mse / reference_mse)
 

def FSS_accuracy_metric(predicted_tensor, target_tensor, half_window_size_px):
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    smoothed_target_tensor = F.conv2d(target_tensor, weight_matrix, padding=half_window_size_px)
    smoothed_prediction_tensor = F.conv2d(predicted_tensor, weight_matrix, padding=half_window_size_px)    
    actual_mse = torch.mean((smoothed_target_tensor - smoothed_prediction_tensor) ** 2)
    reference_mse = torch.mean(smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2)
    return 1. - actual_mse / reference_mse

def FSS_accuracy_metric_gpu(predicted_tensor, target_tensor, half_window_size_px):
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    weight_matrix = weight_matrix.to(device)
    smoothed_target_tensor = F.conv2d(target_tensor, weight_matrix, padding=half_window_size_px)
    smoothed_prediction_tensor = F.conv2d(predicted_tensor, weight_matrix, padding=half_window_size_px)    
    actual_mse = torch.mean((smoothed_target_tensor - smoothed_prediction_tensor) ** 2)
    reference_mse = torch.mean(smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2)
    return 1. - actual_mse / reference_mse


def calc_FSS_for_filter_sizes(predicted_tensor,target_tensor):
    # compute FSS at different scales
    half_window_size=[1, 2, 5, 7, 10, 13, 20, 30, 40, 49]
    ffs_array = []
    FSS_eval_kernel=[]
    for FFS_kernal_size in half_window_size:
        FSS_Score = FSS_accuracy_metric_gpu(predicted_tensor.to(device), target_tensor.to(device),FFS_kernal_size)
        ffs_array.append(FSS_Score.to('cpu')) 
        FSS_eval_kernel.append(FFS_kernal_size*2+1)
    return ffs_array, FSS_eval_kernel

def calc_FSS_for_filter_sizes_HOD(predicted_tensor,target_tensor):
    half_window_size=[2, 10, 20, 40]
    ffs_array = []
    FSS_eval_kernel=[]
    for FFS_kernal_size in half_window_size:
        FSS_Score = FSS_accuracy_metric_gpu(predicted_tensor.to(device), target_tensor.to(device),FFS_kernal_size)
        ffs_array.append(FSS_Score.to('cpu')) 
        FSS_eval_kernel.append(FFS_kernal_size*2+1)
    return ffs_array, FSS_eval_kernel


def brier_skill_score(forecast, reference, observed):
    #Calculate Brier skill score (BSS) between forecast and reference.
    bs_forecast = np.round(metrics.brier_score_loss(observed,forecast),3)
    bs_reference = np.round(metrics.brier_score_loss(observed,reference),3)
    if bs_reference<=0.001:
        return 0
    else:
        return 1 - (bs_forecast / bs_reference)
    
def brier_skill_score_mse(forecast, reference, observed):
    #Calculate Brier skill score (BSS) between forecast and reference.
    bs_forecast = np.round(np.mean((forecast - observed) ** 2, axis=0),3)
    bs_reference = np.round(np.mean((forecast - reference) ** 2, axis=0),3)
    return 1 - (bs_forecast / bs_reference)

def get_previous_date(date_str):
    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Subtract one day using timedelta
    previous_date_obj = date_obj - timedelta(days=1)
    
    # Convert the datetime object back to a string
    previous_date_str = datetime.strftime(previous_date_obj, '%Y-%m-%d')
    
    return previous_date_str

def get_current_date(date_str):
    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Convert the datetime object back to a string
    date_str = datetime.strftime(date_obj, '%Y-%m-%d')
    
    return previous_date_str

def spatial_filter_conv(predicted_image,half_window_size_px):
    
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    weight_matrix=weight_matrix.to(device)
    #half_window_size_px=half_window_size_px #2
    predicted_image=predicted_image.to(device)
    smoothed_predicted_image = F.conv2d(predicted_image, weight_matrix,padding=half_window_size_px)
    return smoothed_predicted_image

def spatial_filter_conv_cpu(predicted_image,half_window_size_px):
    
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    #weight_matrix=weight_matrix.to(device)
    #half_window_size_px=half_window_size_px #2
    #predicted_image=predicted_image.to(device)
    smoothed_predicted_image = F.conv2d(predicted_image, weight_matrix,padding=half_window_size_px)
    return smoothed_predicted_image

def select_model_order(current_month):
    if current_month in [1, 2, 11, 12]: # jan, feb, nov, dec
        return [1, 2, 0]  # SA, EA, WA
    elif current_month in [3, 4]:  # march, apr
        return [2, 1, 0]  # EA, SA, WA
    elif current_month in [5]:  # may
        return [2, 0, 1]  # EA, WA, SA
    elif current_month in [6, 7, 8]:  # jun, jul, aug
        return [0, 2, 1]   # WA, EA, SA
    elif current_month in [9]:  # sep
        return [0, 1, 2]   # WA, SA, EA
    elif current_month in [10]:  # oct
        return [1, 0, 2]   # SA, WA, EA
    else:
        return [0, 1, 2]  # default