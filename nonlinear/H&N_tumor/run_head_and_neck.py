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
import numpy as np
import shutil


#Define sequentially jobs running

def run(jobname, Cpus, Domains, Gpus):
    """
    Submit to Abaqus solver and run the job. 
    """

    mdb.JobFromInputFile(name=jobname, inputFileName=jobname, type=ANALYSIS, atTime=None, 
        waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, 
        userSubroutine='', scratch='', multiprocessingMode=DEFAULT, numCpus=Cpus, numDomains=Domains, numGPUs=Gpus)
    mdb.jobs[jobname].submit()
    mdb.jobs[jobname].waitForCompletion()


def collectFiles(directory, jobname, extension, target_folder):
    """
    Move files of the specified task to corresponding folders. 

    Parameters: 
    ----------
        directory: String. 
            The working directory of Abaqus. 
        jobname: String. 
            The name of the job/task. 
            If the job is not completed/aborted, then the extension "_aborted" will be addressed at the end of the jobname. 
        extension: String. 
            The format of the file. 
            Starting with '.'. 
        target_folder: String. 
            The target subfolder of the to-be-moved file in the directory.
    
    Returns:
    ----------
        target_path: String. 
            The path of the file after relocating. 
            When the file does not exist, return "".
    """

    source_path = os.path.join(directory, jobname+extension)
    target_path = os.path.join(directory, target_folder, jobname+extension)

    if os.path.exists(source_path):
        shutil.move(source_path, target_path)
        return target_path
    
    else: return ""


# ***************************************************************************************************** #
# transferred_data = np.load("training_parameters_transfer.npy")
# current_directory = transferred_data.item().get("current_directory")
# inp_folder = transferred_data.item().get("inp_folder")
# stress_folder_name = transferred_data.item().get("results_folder_path_stress")
# coord_folder_name = transferred_data.item().get("results_folder_path_coor")

current_directory = "C:/Users/13426/Desktop/soft_tissue_tracking/code/ANN/nonlinear/head_and_neck" # Directory with only input files of stage1 and stage2. 
inp_folder = "inp_interpolation"
working_directory = os.path.join(current_directory, inp_folder)
os.chdir(working_directory)
file_list = [file for file in os.listdir(working_directory) if os.path.isdir(file) == 0]

stress_folder_name = "stress"
coord_folder_name = "coor"
fil_folder_name = "fil"
odb_folder_name = "odb"
sta_folder_name = "sta"
log_folder_name = "log"
inp_folder_name = "inp"

if not os.path.isdir(stress_folder_name): os.mkdir(os.path.join(working_directory, stress_folder_name))
if not os.path.isdir(coord_folder_name): os.mkdir(os.path.join(working_directory, coord_folder_name))
if not os.path.isdir(fil_folder_name): os.mkdir(os.path.join(working_directory, fil_folder_name))
if not os.path.isdir(odb_folder_name): os.mkdir(os.path.join(working_directory, odb_folder_name))
if not os.path.isdir(sta_folder_name): os.mkdir(os.path.join(working_directory, sta_folder_name))
if not os.path.isdir(log_folder_name): os.mkdir(os.path.join(working_directory, log_folder_name))
if not os.path.isdir(inp_folder_name): os.mkdir(os.path.join(working_directory, inp_folder_name))

Cpus = 8
Domains = 8
Gpus = 0
time_break = 5 # The break time to let Abaqus finish processing all files. Unit: s. 

string_list = ["CPU processors: {}".format(Cpus),
               "Domains: {}".format(Domains),
               "GPU processors: {}".format(Gpus),
               "----------------------------------------------------------"]
time_total = 0 # Float. The total time spent for all simulations. 

for file_name in file_list:
    if file_name.split('.')[-1] != "inp": continue

    start_time = time.time()
    jobname = file_name.split('.')[0]
    run(jobname, Cpus, Domains, Gpus)
    end_time_job = time.time()
    time.sleep(time_break)

    sta_path = "{}/{}.sta".format(working_directory, jobname) # Path to the corresponding .sta file

    if not os.path.exists(sta_path):
        time.sleep(time_break)
        check_list_temp = [file for file in os.listdir(working_directory) if file.split('.')[0] == jobname]

        for file in check_list_temp:
            if (os.path.exists(os.path.join(working_directory, file)) and 
                file.split('.')[-1] != "inp" and
                file.split('.')[-1] != "log" and
                file.split('.')[-1] != "odb" and
                file.split('.')[-1] != "sta"): 
                os.remove(os.path.join(working_directory, file))

            if (file.split('.')[-1] == "inp" or
                file.split('.')[-1] == "log" or
                file.split('.')[-1] == "odb" or
                file.split('.')[-1] == "sta"): 
                ori_name = os.path.join(working_directory, file)
                dst_name = ori_name.split('.')[0] + "_aborted." + file.split('.')[-1]
                os.rename(ori_name, dst_name)
        
        jobname = file_name.split('.')[0] + "_aborted"

    else: 
        with open(sta_path, "rt") as f: lines = f.read().splitlines()

        if not lines[-1] == " THE ANALYSIS HAS COMPLETED SUCCESSFULLY":
            time.sleep(time_break)
            check_list_temp = [file for file in os.listdir(working_directory) if file.split('.')[0] == jobname]

            for file in check_list_temp:
                if (os.path.exists(os.path.join(working_directory, file)) and 
                    file.split('.')[-1] != "inp" and
                    file.split('.')[-1] != "log" and
                    file.split('.')[-1] != "odb" and
                    file.split('.')[-1] != "sta"): 
                    os.remove(os.path.join(working_directory, file))

                if (file.split('.')[-1] == "inp" or
                    file.split('.')[-1] == "log" or
                    file.split('.')[-1] == "odb" or
                    file.split('.')[-1] == "sta"): 
                    ori_name = os.path.join(working_directory, file)
                    dst_name = ori_name.split('.')[0] + "_aborted." + file.split('.')[-1]
                    os.rename(ori_name, dst_name)
            
            jobname = file_name.split('.')[0] + "_aborted"
        
        else: 
            check_list_temp = [file for file in os.listdir(working_directory) if file.split('.')[0] == jobname]

            for file in check_list_temp:
                if (os.path.exists(os.path.join(working_directory, file)) and 
                    file.split('.')[-1] != "inp" and
                    file.split('.')[-1] != "log" and
                    file.split('.')[-1] != "fil" and
                    file.split('.')[-1] != "sta" and
                    file.split('.')[-1] != "odb"): 
                    os.remove(os.path.join(working_directory, file))
    
    target_path_fil_temp = collectFiles(working_directory, jobname, ".fil", fil_folder_name)
    _ = collectFiles(working_directory, jobname, ".sta", sta_folder_name)
    _ = collectFiles(working_directory, jobname, ".inp", inp_folder_name)
    _ = collectFiles(working_directory, jobname, ".log", log_folder_name)
    target_path_odb_temp = collectFiles(working_directory, jobname, ".odb", odb_folder_name)

    end_time_total = time.time()
    elapsed_time_run = end_time_job - start_time
    elapsed_time_total = end_time_total - start_time
    time_total += elapsed_time_total

    if target_path_fil_temp == "": isFil_exist = "False"
    else: 
        if os.path.exists(target_path_fil_temp): isFil_exist = "True"
        else: isFil_exist = "False"

    if jobname.split('_')[-1] == "aborted": status_string = "Aborted"
    else: status_string = "Completed"

    print_string_temp = ("Job: " + jobname.split('_')[0] + " | File: " + jobname.split('_')[0]+".inp" + 
                    " | Status: " + status_string + " | Fil: " + isFil_exist + " | Run time: %.4f s" % (elapsed_time_run) + 
                    " | Total time: %.4f s" % (elapsed_time_total))

    string_list.append(print_string_temp)
    content = '\n'.join(string_list)
    with open("simulation.log", 'w') as f: f.write(content)

    if target_path_odb_temp != "" and np.random.rand() <= 0.7: os.remove(target_path_odb_temp)
    else: continue

string_list += ["Total time spent: {} hrs".format(time_total / 3600.0)]
content = '\n'.join(string_list)
with open("simulation.log", 'w') as f: f.write(content)
