# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 13:08:16 2020

@author: haolinl
"""

import copy
import os

import numpy as np
import scipy.io # For extracting data from .mat file
import shutil

transfer_data_mat = scipy.io.savemat("training_parameters_transfer.mat")

current_directory = transfer_data_mat["current_directory"][0]
inp_folder = transfer_data_mat["inp_folder"][0]
working_directory = os.path.join(current_directory, inp_folder)
file_list = os.path.listdir(working_directory)

for file in file_list: 
    if os.path.isdir(file): continue

    source_path = os.path.join(current_directory, file)
    fil_folder_name = "fil"
    odb_folder_name = "odb"
    sta_folder_name = "sta"
    log_folder_name = "log"
    inp_folder_name = "inp"

    if file.split('.')[-1] == "fil": 
        shutil.move(source_path, os.path.join(current_directory, fil_folder_name, file))
    elif file.split('.')[-1] == "odb": 
        shutil.move(source_path, os.path.join(current_directory, odb_folder_name, file))
    elif file.split('.')[-1] == "sta": 
        shutil.move(source_path, os.path.join(current_directory, sta_folder_name, file))
    elif file.split('.')[-1] == "log": 
        shutil.move(source_path, os.path.join(current_directory, log_folder_name, file))
    elif file.split('.')[-1] == "inp": 
        shutil.move(source_path, os.path.join(current_directory, inp_folder_name, file))
    else: continue