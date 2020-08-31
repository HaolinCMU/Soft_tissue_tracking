# Last modified: 1/16/2019 1:37AM - Haolin

# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__

import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import os
import time
#import shutil

#Define sequentially jobs running

def run(jobname, Cpus, Domains, Gpus):
    mdb.JobFromInputFile(name=jobname, inputFileName=jobname, type=ANALYSIS, atTime=None, 
        waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, 
        userSubroutine='', scratch='', multiprocessingMode=DEFAULT, numCpus=Cpus, numDomains=Domains, numGPUs=Gpus)
    mdb.jobs[jobname].submit()
    mdb.jobs[jobname].waitForCompletion()


# ***************************************************************************************************** #

current_directory = "C:/Users/13426/Desktop/soft_tissue_tracking/code/ANN/nonlinear" # Directory with only input files of stage1 and stage2. 
inp_folder = "inp_files"
stress_folder_name = "stress"
coord_folder_name = "coor"
working_directory = os.path.join(current_directory, inp_folder)
os.chdir(working_directory)
file_list = [file for file in os.listdir(working_directory) if os.path.isdir(file) == 0]

Cpus = 4
Domains = 4
Gpus = 0
time_break = 5 # The break time to let Abaqus finish processing all files. Unit: s. 

stage1_list, stage2_list = [], []

for file_name in file_list:
    if file_name.split('.')[-1] != "inp": continue

    jobname = file_name.split('.')[0]
    run(jobname, Cpus, Domains, Gpus)
    time.sleep(time_break)

    sta_path = "{}/{}.sta".format(working_directory, jobname) # Path to the corresponding .sta file

    if not os.path.exists(sta_path):
        time.sleep(time_break)
        check_list_temp = [file for file in os.listdir(working_directory) if file.split('.')[0] == jobname]

        for file in check_list_temp:
            if (os.path.exists(os.path.join(working_directory, file)) and 
                file.split('.')[-1] != "inp" and
                file.split('.')[-1] != "log"): 
                os.remove(os.path.join(working_directory, file))

            if (file.split('.')[-1] == "inp"): 
                ori_name = os.path.join(working_directory, file)
                dst_name = ori_name.split('.')[0] + "_(aborted).inp"
                os.rename(ori_name, dst_name)
        
        continue

    with open(sta_path, "rt") as f: lines = f.read().splitlines()

    if not lines[-1] == " THE ANALYSIS HAS COMPLETED SUCCESSFULLY":
        time.sleep(time_break)
        check_list_temp = [file for file in os.listdir(working_directory) if file.split('.')[0] == jobname]

        for file in check_list_temp:
            if (os.path.exists(os.path.join(working_directory, file)) and 
                file.split('.')[-1] != "inp" and
                file.split('.')[-1] != "log"): 
                os.remove(os.path.join(working_directory, file))

            if (file.split('.')[-1] == "inp"): 
                ori_name = os.path.join(working_directory, file)
                dst_name = ori_name.split('.')[0] + "_(aborted).inp"
                os.rename(ori_name, dst_name)
        
        continue

    check_list_temp = [file for file in os.listdir(working_directory) if file.split('.')[0] == jobname]

    for file in check_list_temp:
        if (os.path.exists(os.path.join(working_directory, file)) and 
            file.split('.')[-1] != "inp" and
            file.split('.')[-1] != "log" and
            file.split('.')[-1] != "fil" and
            file.split('.')[-1] != "odb"): 
            os.remove(os.path.join(working_directory, file))

if not os.path.isdir(stress_folder_name): os.mkdir(os.path.join(working_directory, stress_folder_name))
if not os.path.isdir(coord_folder_name): os.mkdir(os.path.join(working_directory, coord_folder_name))
