# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:00:55 2020

@author: haolinl
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io # For extracting data from .mat file


figure_folder_path = 'figure' # The directory of figure folder. 

if not os.path.isdir(figure_folder_path): os.mkdir(figure_folder_path)

data_file = scipy.io.loadmat('ANN_benchmark_results.mat')
max_nodal_error_testPCA = data_file['max_nodal_error_testPCA']
mean_nodal_error_testPCA = data_file['mean_nodal_error_testPCA']
max_nodal_error = data_file['max_nodal_error']
mean_nodal_error = data_file['mean_nodal_error']
eigVal_full, eigVal = data_file['eigVal_full'], data_file['eigVal']
eigVect_full, eigVect = data_file['eigVect_full'], data_file['eigVect']
test_deformation_label = data_file['test_deformation_label']
dist_nodal_matrix = data_file['dist_nodal_matrix']
dist_nodal_matrix_testPCA = data_file['dist_nodal_matrix_testPCA']
FM_num, PC_num = data_file['FM_num'], data_file['PC_num']
FM_indices = data_file['FM_indices']


# Plot & Calculate max & mean error parameters
max_median_testPCA = np.median(max_nodal_error_testPCA)
max_max_testPCA = np.max(max_nodal_error_testPCA)
max_min_testPCA = np.min(max_nodal_error_testPCA)

mean_median_testPCA = np.median(mean_nodal_error_testPCA)
mean_max_testPCA = np.max(mean_nodal_error_testPCA)
mean_min_testPCA = np.min(mean_nodal_error_testPCA)

max_median = np.median(max_nodal_error)
max_max = np.max(max_nodal_error)
max_min = np.min(max_nodal_error)

mean_median = np.median(mean_nodal_error)
mean_max = np.max(mean_nodal_error)
mean_min = np.min(mean_nodal_error)

print(FM_indices)


# Box plot
# ANN reconstruction
fig, axes = plt.subplots(1,1,figsize = (20, 12.8))
plt.rcParams.update({"font.size": 35})
plt.tick_params(labelsize=35)
df = pd.DataFrame(dist_nodal_matrix[:,0:12], # The first 12 samples.
                  columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
f = df.boxplot(sym = 'o',            
           vert = True,
           whis=1.5, 
           patch_artist = True,
           meanline = False,showmeans = True,
           showbox = True,
           showfliers = True,
           notch = False,
           return_type='dict')
plt.xlabel('Samples (x12) - ANN Reconstruction', fontsize=40)
plt.ylabel('Euclidean distance (mm)', fontsize=40)
plt.savefig(figure_folder_path + '/boxPlot_ANN.png')


# PCA reconstruction
fig, axes = plt.subplots(1,1,figsize = (20, 12.8))
plt.rcParams.update({"font.size": 35})
plt.tick_params(labelsize=35)
df = pd.DataFrame(dist_nodal_matrix_testPCA[:,0:12], # The first 12 samples.
                  columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
f = df.boxplot(sym = 'o',            
           vert = True,
           whis=1.5, 
           patch_artist = True,
           meanline = False,showmeans = True,
           showbox = True,
           showfliers = True,
           notch = False,
           return_type='dict')
plt.xlabel('Samples (x12) - PCA Reconstruction', fontsize=40)
plt.ylabel('Euclidean distance (mm)', fontsize=40)
plt.savefig(figure_folder_path + '/boxPlot_PCA.png')


# Plot PCA curve
x_plot = range(50)
y_plot = np.real(eigVal_full[0,0:50]).reshape(-1,1)
plt.figure(figsize=(20.0, 12.8))
plt.rcParams.update({"font.size": 35})
plt.tick_params(labelsize=35)
plt.plot(x_plot, y_plot, linewidth=3.0, linestyle='-', marker='*', 
            color='b', markersize=7.0, label='Eigen Values')
plt.axvline(PC_num, 0, 1)
plt.legend(loc='upper right', prop={'size': 40})
plt.xlabel('Principal Components', fontsize=40)
plt.ylabel('Variance', fontsize=40)
plt.title('PCA performance')
plt.savefig(figure_folder_path + '/PCA_curve.png')


# Boxplot mean & max error distribution among all testing samples
fig, axes = plt.subplots(1, 1, figsize = (20, 12.8))
plt.rcParams.update({"font.size": 35})
plt.tick_params(labelsize=35)
df = pd.DataFrame(mean_nodal_error_testPCA,
                  columns=['mean nodal error of testing samples (pure PCA)'])
f = df.boxplot(sym = 'o',            
           vert = True,
           whis=1.5, 
           patch_artist = True,
           meanline = False,showmeans = True,
           showbox = True,
           showfliers = True,
           notch = False,
           return_type='dict')
plt.ylabel('Euclidean distance (mm)', fontsize=40)
plt.savefig(figure_folder_path + '/boxPlot_meanError_PCA.png')

fig, axes = plt.subplots(1, 1, figsize = (20, 12.8))
plt.rcParams.update({"font.size": 35})
plt.tick_params(labelsize=35)
df = pd.DataFrame(max_nodal_error_testPCA,
                  columns=['max nodal error of testing samples (pure PCA)'])
f = df.boxplot(sym = 'o',            
           vert = True,
           whis=1.5, 
           patch_artist = True,
           meanline = False,showmeans = True,
           showbox = True,
           showfliers = True,
           notch = False,
           return_type='dict')
plt.ylabel('Euclidean distance (mm)', fontsize=40)
plt.savefig(figure_folder_path + '/boxPlot_maxError_PCA.png')

fig, axes = plt.subplots(1, 1, figsize = (20, 12.8))
plt.rcParams.update({"font.size": 35})
plt.tick_params(labelsize=35)
df = pd.DataFrame(max_nodal_error,
                  columns=['max nodal error of testing samples'])
f = df.boxplot(sym = 'o',            
           vert = True,
           whis=1.5, 
           patch_artist = True,
           meanline = False,showmeans = True,
           showbox = True,
           showfliers = True,
           notch = False,
           return_type='dict')
plt.ylabel('Euclidean distance (mm)', fontsize=40)
plt.savefig(figure_folder_path + '/boxPlot_maxError.png')

fig, axes = plt.subplots(1, 1, figsize = (20, 12.8))
plt.rcParams.update({"font.size": 35})
plt.tick_params(labelsize=35)
df = pd.DataFrame(mean_nodal_error,
                  columns=['mean nodal error of testing samples'])
f = df.boxplot(sym = 'o',            
           vert = True,
           whis=1.5, 
           patch_artist = True,
           meanline = False,showmeans = True,
           showbox = True,
           showfliers = True,
           notch = False,
           return_type='dict')
plt.ylabel('Euclidean distance (mm)', fontsize=40)
plt.savefig(figure_folder_path + '/boxPlot_meanError.png')
