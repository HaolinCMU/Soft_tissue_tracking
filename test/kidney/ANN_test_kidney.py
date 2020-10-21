# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 22:51:16 2020

@author: haolinl
"""

import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io # For extracting data from .mat file
import scipy.stats as st
import torch
import torch.nn as nn
import torch.utils.data


class Net1(nn.Module):
    """
    MLP modeling hyperparams:
    ----------
        Input: FM_num (FMs' displacements). 
        Hidden layers: Default architecture: 128 x 64 (From Haolin and Houriyeh). Optimization available. 
        Output: PC_num (weights generated from deformation's PCA). 
    """
    
    def __init__(self, FM_num, PC_num):
        """
        Parameters:
        ----------
            FM_num: Int. 
                The number of fiducial markers. 
            PC_num: Int. 
                The number of picked principal compoments. 
        """
        
        super(Net1, self).__init__()
        self.FM_num = FM_num
        self.PC_num = PC_num
        self.hidden_1 = nn.Sequential(
            nn.Linear(int(self.FM_num*3), 128),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.hidden_2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.out_layer = nn.Linear(64, self.PC_num)
        
        
    def forward(self, x):
        """
        Forward mapping: FM displacements -> Principal weights. 

        Parameters:
        ----------
            x: 2D Array. 
                Matrix of FM displacements of all DOFs. 

        Returns:
        ----------
            output: 2D Array. 
                Matrix of principal weights. 
        """
        
        f1 = self.hidden_1(x)
        f2 = self.hidden_2(f1)
        output = self.out_layer(f2)
        return output


def readFile(file_path):
    """
    Read the file (mostly, should be .csv files of deformed coordinates) and return the list of string lines. 

    Parameters:
    ----------
    file_path: String. 
        The path of the target redable file. 
    
    Returns:
    ----------
    lines: List of strings. 
        Lines of the file's content. 
    """

    with open(file_path, 'rt') as f: lines = f.read().splitlines()

    return lines


def normalization(data):
    """
    Normalize the input data (displacements) of each direction within the range of [0,1].
    For unmatched displacement range along with different directions/each single feature.

    Parameters:
    ----------
        data: 2D Array. 
            Matrix of training/testing input data. 

    Returns:
    ----------
        data_nor: 2D Array. 
            Matrix of the normalized data with the same shape as the input. 
        norm_params: 1D Array (6 x 1). 
            Containing "min" and "max"  of each direction for reconstruction. 
            Row order: [x_min; x_max; y_min; y_max; z_min; z_max]. 
    """
    
    data_nor, norm_params = np.zeros(data.shape), None
    
    # Partition the data matrix into x,y,z_matrices
    x_temp, y_temp, z_temp = data[::3,:], data[1::3,:], data[2::3,:]
    x_max, x_min = np.max(x_temp), np.min(x_temp)
    y_max, y_min = np.max(y_temp), np.min(y_temp)
    z_max, z_min = np.max(z_temp), np.min(z_temp)
    
    # Min-max normalization: [0,1]
    x_temp = (x_temp - x_min) / (x_max - x_min)
    y_temp = (y_temp - y_min) / (y_max - y_min)
    z_temp = (z_temp - z_min) / (z_max - z_min)
    
    data_nor[::3,:], data_nor[1::3,:], data_nor[2::3,:] = x_temp, y_temp, z_temp
    norm_params = np.array([x_max, x_min, y_max, y_min, z_max, z_min]).astype(float).reshape(-1,1)
    
    return data_nor, norm_params


def matrixShrink(data_matrix, fix_indices_list=[]):
    """
    Remove rows of zero displacement (fixed DOFs).

    Parameters:
    ----------
        data_matrix: 2D Array. 
            Size: nDOF x SampleNum. 
            The full matrix of deformation data.
        fix_indices_list (optional): List of ints.
            The list of fixed indices. 
            Indexed from 1. 
            For nonlinear dataset, this list should be specified. 
            Default: []. 

    Returns:
    ----------
        data_shrinked: 2D Array. 
            Size: nDOF' x SampleNum. 
            The matrix without zero rows.
        nDOF: Int. 
            Number of all DOFs of original deformation matrix. 
        non_zero_indices_list: List. 
            All indices of non zero rows for deformation reconstruction. 
            In exact order.  
    """
    
    if fix_indices_list == []:
        nDOF = data_matrix.shape[0]
        zero_indices_list, non_zero_indices_list = [], []

        for i in range(nDOF):
            if data_matrix[i,0] == 0: zero_indices_list.append(i)
            else: non_zero_indices_list.append(i)
        
        data_shrinked = np.delete(data_matrix, zero_indices_list, axis=0)

    else:
        fix_indices_list = [item-1 for item in fix_indices_list] # Make the fixed nodes indexed from 0. 
        nDOF = data_matrix.shape[0]
        zero_indices_list, non_zero_indices_list = [], []

        for i in range(int(nDOF/3)): # Iterate within the range of node_num. 
            if i in fix_indices_list: 
                zero_indices_list.append(i*3)
                zero_indices_list.append(i*3+1)
                zero_indices_list.append(i*3+2)
            else: 
                non_zero_indices_list.append(i*3)
                non_zero_indices_list.append(i*3+1)
                non_zero_indices_list.append(i*3+2)

        data_shrinked = np.delete(data_matrix, zero_indices_list, axis=0)
    
    return data_shrinked, nDOF, non_zero_indices_list
    

def zeroMean(data_matrix, training_ratio, mean_vect_input=[]):
    """
    Shift the origin of new basis coordinate system to mean point of the training data. 

    Parameters:
    ----------
        data_matrix: 2D Array. 
            Size: nFeatures x nSamples.
        training_ratio: float.
            The ratio of training dataset. 
        mean_vect_input (optional): 1D List of floats. 
            The user input of mean vector of training dataset. 
            Default: [].

    Returns:
    ----------
        data_new: 2D Array with the same size as data_matrix. 
            Mean-shifted data. 
        mean_vect: 1D Array of float. 
            The mean value of each feature. 
    """
    
    if mean_vect_input == []:
        training_index = int(np.ceil(data_matrix.shape[1] * training_ratio)) # Samples along with axis-1.
        mean_vect = np.mean(data_matrix[:,0:training_index], axis=1) # Compute mean along with sample's axis. 
    else:
        mean_vect = np.array(mean_vect_input).astype(float).reshape(-1,)

    data_new = np.zeros(data_matrix.shape)

    for i in range(data_matrix.shape[1]):
        data_new[:,i] = data_matrix[:,i] - mean_vect
    
    return data_new, mean_vect


def PCA(data_matrix, PC_num, training_ratio, bool_PC_norm=False):
    """
    Implement PCA on tumor's deformation covariance matrix (Encoder) - training set. 

    Parameters:
    ----------
        data_matrix: 2D Array. 
            Size: nNodes*3 x SampleNum. 
            Each DOF is a feature. Mean-shifted.  
        PC_num: Int. 
            The number of picked PCs.
        training_ratio: float.
            The ratio of training dataset.
        bool_PC_norm (optional): Boolean.
            True: turn on the eigenvector normalization of principal eigenvectors. 
            False: turn off the eigenvector normalization of principal eigenvectors. 
            Default: False.

    Returns:
    ----------
        eigVect_full: 2D Array. 
            Size: nNodes*3 x nNodes*3. 
            All principal eigen-vectors.
        eigVal_full: 1D Array. 
            Size: nNodes*3 x 1. 
            All principal eigen-values. 
        eigVect: 2D Array. 
            Size: nNodes*3 x PC_num. 
            Principal eigen-vectors.
        eigVal: 1D Array. 
            Size: PC_num x 1. 
            Principal eigen-values. 
        weights: 2D Array (complex). 
            Size: PC_num x SampleNum. 
            Projected coordinates on each PC of all samples. 
    """
    
    # Compute covariance matrix & Eigendecompostion
    training_index = int(np.ceil(data_matrix.shape[1] * training_ratio)) # Samples along with axis-1.
    cov_matrix = data_matrix[:,0:training_index] @ np.transpose(data_matrix[:,0:training_index]) # Size: nDOF * nDOF
    eigVal_full, eigVect_full = np.linalg.eig(cov_matrix)
    
    # # Eigencomponent-wise normalization. 
    # eigVect_full = eigVect_full / eigVal_full
    
    # PCA
    eigVal, eigVect = np.zeros(shape=(PC_num, 1), dtype=complex), np.zeros(shape=(eigVect_full.shape[0], PC_num), dtype=complex)
    eigVal_sorted_indices = np.argsort(np.real(eigVal_full))
    eigVal_PC_indices = eigVal_sorted_indices[-1:-(PC_num+1):-1] # Pick PC_num indices of largest principal eigenvalues
    
    for i, index in enumerate(eigVal_PC_indices): # From biggest to smallest
        eigVal[i,0] = eigVal_full[index] # Pick PC_num principal eigenvalues. Sorted. 
        eigVect[:,i] = eigVect_full[:,index] # Pick PC_num principal eigenvectors. Sorted. 
    
    # Compute weights of each sample on the picked basis (encoding). 
    if bool_PC_norm: weights = np.transpose(eigVect/eigVal.reshape(-1,)) @ data_matrix # Size: PC_num * SampleNum, complex. 
    else: weights = np.transpose(eigVect) @ data_matrix # Size: PC_num * SampleNum, complex.
    
    return eigVect_full, eigVal_full, eigVect, eigVal, weights


def dataReconstruction(eigVect, eigVal, weights, mean_vect, nDOF, non_zero_indices_list, bool_PC_norm=False):
    """
    Reconstruct the data with eigenvectors and weights (Decoder). 

    Parameters:
    ----------
        eigVect: 2D Array. 
            Size: nDOF x PC_num. 
            Principal eigenvectors aligned along with axis-1. 
        eigVal: 1D Array.
            Size: PC_num x 1.
            Principal eigenvalues corresponding to the columns of principal eigenvectors.
        weights: 2D Array (complex). 
            Size: PC_num x SampleNum. 
            Weights of each sample aligned along with axis-1.
        mean_vect: 1D Array. 
            The mean value of each feature of training data. 
        nDOF: Int. 
            Number of all DOFs of original deformation matrix. 
        non_zero_indices_list: List. 
            All indices of non zero rows for deformation reconstruction. 
        bool_PC_norm (optional): Boolean.
            True: turn on the eigenvector normalization of principal eigenvectors. 
            False: turn off the eigenvector normalization of principal eigenvectors. 
            Default: False.

    Returns:
    ----------
        data_reconstruct: 2D Array. 
            Size: nDOF x SampleNum. 
            Reconstructed deformation results. 
    """
    
    # Transform weights back to original vector space (decoding)
    if bool_PC_norm: data_temp = (eigVect * eigVal.reshape(-1,)) @ weights
    else: data_temp = eigVect @ weights

    for i in range(data_temp.shape[1]):
        data_temp[:,i] += mean_vect # Shifting back
    
    data_reconstruct = np.zeros(shape=(nDOF, data_temp.shape[1]), dtype=complex)

    for i, index in enumerate(non_zero_indices_list):
        data_reconstruct[index,:] = data_temp[i,:]
    
    return np.real(data_reconstruct)


def dataProcessing(data_x, data_y, batch_size, FM_indices, bool_norm=False):
    """
    Data preprocessing. 

    Parameters:
    ----------
        data_x: 2D Array (nDOF x SampleNum). 
            The deformation data (x SampleNum) of all DOFs. 
        data_y: 2D Array (PC_num x SampleNum, complex). 
            The label data (x SampleNum). 
            Here it should be the weights vectors for the force field reconstruction. 
        batch_size: Int. 
            The size of a single training batch input.
        FM_indices: 1D Array. 
            Randomly picked FM indices. 
            Typical size: 5. 
        bool_norm (optional): Boolean. 
            True: conduct directional input normalization. 
            False: skip directional input normalization. 
            Default: False.

    Returns:
    ----------
        test_dataloader: Tensor dataloader. 
            Testing dataset.
        norm_params: 1D Array. 
            Min and max values of data matrix. 
            Return empty list if bool_norm == 0. 
    """
    
    # Data normalization
    if bool_norm: data_x, norm_params = normalization(data_x)
    else: norm_params = []
    
    data_x_FM = np.zeros(shape=(int(len(FM_indices)*3), data_x.shape[1]))
    for i, index in enumerate(FM_indices):
        data_x_FM[i*3:(i+1)*3,:] = data_x[int(index*3):int((index+1)*3),:]
    data_x = copy.deepcopy(data_x_FM) # Size: FM_num*3 x SampleNum
    data_y = np.real(data_y) # Discard imaginary part of the weights for the convenience of training. 
    
    # Partition the whole dataset into "train" and "test". 
    test_x = torch.from_numpy(data_x).float() # size: 15 x nTrain
    test_y = torch.from_numpy(data_y).float() # size: 20 x nTrain
    
    # Generate dataloaders 
    # Make sure the sample dimension is on axis-0. 
    test_dataset = torch.utils.data.TensorDataset(np.transpose(test_x), 
                                                  np.transpose(test_y))
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset
    )
    
    return test_dataloader, norm_params


def testNet(test_dataloader, neural_net, device):
    """
    MLP testing. 

    Parameters:
    ----------
        test_dataloader: Tensor dataloader. 
            Testing dataset.
        neural_net: Pre-trained MLP. 
        device: CPU/GPU. 
    
    Returns:
    ----------
        pred_y_list: List of vectors. 
            The results of predictions. 
        test_y_list: List of vectors. 
            The results of original labels(weights). 
        lossList_test: List of floats. 
            The prediction error (MSE) of each test sample (weights).  
    """
    
    loss = nn.MSELoss()
    pred_y_List, test_y_List, lossList_test = [], [], [] # List of predicted vector, test_y and loss of each sample. 
    
    for (displacements, weights) in test_dataloader:
        x_sample = torch.autograd.Variable(displacements)
        x_sample = x_sample.to(device)
        weights = weights.to(device)
        pred_y = neural_net(x_sample)
        pred_y_List.append(np.array(pred_y.cpu().data.numpy()).astype(float).reshape(-1,1))
        test_y_List.append(np.array(weights.cpu().data.numpy()).astype(float).reshape(-1,1))
        loss_test_temp = loss(pred_y, weights)
        lossList_test.append(loss_test_temp.cpu().data.numpy())
    
    return pred_y_List, test_y_List, lossList_test


def deformationExtraction(data_mat, variable_name, original_node_number, loads_num, results_folder_path, alpha_indexing_vector=[]):
    """
    Extract deformation information from original configuration (.mat file) and deformed configuration (.csv file). 

    Parameters:
    ----------
        data_mat: mat file content. 
            The .mat file containing the node list of original configuration. 
        variable_name: String. 
            The variable name of the node list. 
        original_node_number: Int. 
            The number of original nodes. Excluding edge's midpoints. 
        loads_num: Int. 
            The number of couple regions = number of rf points = the header legnth of the coordinate .csv file, 
        results_folder_path: String. 
            The path of the directory containing the result .csv files. 
        alpha_indexing_vector (optional): List of floats.
            The vector containing all alphas for linear interpolation of force fields.
            Default: []. 
    
    Returns:
    ----------
        data_x: 2D Array of float. 
            Size: node_num*3 x Sample_num (file_num). 
            The matrix of deformation of all nodes. 
    """

    orig_config_temp = data_mat[variable_name] # Float matrix. Extract the node-coord data of the original configuration. 
    orig_config_temp = orig_config_temp.astype(float).reshape(-1,1) # Size: node_num*3 x 1. Concatenated as xyzxyz...

    deformed_config_file_list = [file for file in os.listdir(results_folder_path) if not os.path.isdir(file) and file.split('.')[-1] == "csv"]
    data_x, alpha_vector = None, []

    for index, file in enumerate(deformed_config_file_list):
        if alpha_indexing_vector != []:
            file_number = int(file.split('_')[0])
            alpha_vector.append(alpha_indexing_vector[file_number-20001]) # 20001: from "nonlinearCasesCreation.py". Change it with the settings in "nonlinearCasesCreation.py". 

        lines_temp = readFile(os.path.join(results_folder_path, file))
        nodes_list_temp = []

        for line in lines_temp[loads_num:original_node_number+loads_num]:
            coord_list_temp = [float(num) for num in line.split(',')[1:]]
            nodes_list_temp.append(copy.deepcopy(coord_list_temp))
        
        deformed_config_temp = np.array(nodes_list_temp).astype(float).reshape(-1,1) # Size: node_num*3 x 1. Concatenated as xyzxyz...
        x_temp = deformed_config_temp - orig_config_temp # 1D Array. Size: node_num*3 x 1. Calculate the deformation. Unit: m. 

        if index == 0: data_x = copy.deepcopy(x_temp)
        else: data_x = np.hstack((data_x, copy.deepcopy(x_temp)))
    
    alpha_vector = np.array(alpha_vector).astype(float).reshape(-1,)
    
    return data_x, alpha_vector


def main():
    """
    ANN testing for external unseen nonlinear dataset. 

    Pipeline: 
    ----------
        1. Run "ANN_nonlinear_validation.py", and obtain the result files of "ANN_benchmark_results.mat", "ANN_trained.pkl", as well as "data.mat" for specific models. 
        2. Copy & Past the above three files to the same directory of "ANN_test.py", as well sa the dataset used for testing to the folder named "data".
        3. Double check if the ANN srtructure is the same as the one used in "ANN_nonlinear_validation.py". 
        4. Run "ANN_test.py". 
        5. Run "resultPlot.py". 
        6. Run "visualizeResults.m" for deformation reconstruction visualization. 
    """

    result_mat_file_name = "ANN_benchmark_results.mat"
    mesh_mat_file_name = "data_kidney.mat"
    result_mat = scipy.io.loadmat(result_mat_file_name)
    mesh_mat = scipy.io.loadmat(mesh_mat_file_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FM_num, PC_num = result_mat["FM_num"][0,0], result_mat["PC_num"][0,0]
    eigVal_full, eigVal = result_mat['eigVal_full'], result_mat['eigVal']
    eigVect_full, eigVect = result_mat['eigVect_full'], result_mat['eigVect']
    mean_vect_list = list(result_mat["mean_vect"])
    center_indices_list = result_mat["center_indices"] - 1
    FM_indices = result_mat["FM_indices"] - 1

    batch_size = result_mat["batch_size"] 
    isNormOn = False # True/Flase: Normalization on/off.
    isKCenter = True # True/Flase: Y/N for implementing optimized k-center. 
    isPCNormOn = result_mat['isPCNormOn'][0,0]

    ANN_model_file_name = "ANN_trained.pkl"
    result_folder_path = "data"

    orig_config_var_name, fix_node_list_name = "NodeI", "fix_node_list"
    
    neural_net = torch.load(ANN_model_file_name)
    neural_net.eval()
    
    # neural_net = Net1(FM_num, PC_num).to(device)
    # neural_net.load_state_dict(torch.load(ANN_model_file_name))
    # neural_net.eval()

    original_node_number, loads_num = mesh_mat[orig_config_var_name].shape[0], 0 # loads_num: when using coupling constraints, a number of assembly nodes should be specifiled. 
    fix_indices_list = list(result_mat[fix_node_list_name][0]) # List of ints. The list of fixed node indices. Indexed from 1. From "nonlinearCasesCreation.py". Default: None.
    alphaIndexingVector = list(result_mat["alpha_indexing_vector"]) # List of floats. The alphas for interpolated data. 
    test_data, alpha_vector = deformationExtraction(mesh_mat, orig_config_var_name, original_node_number, loads_num, result_folder_path, 
                                                    alpha_indexing_vector=alphaIndexingVector) # change the variable's name if necessary. 

    data_x, nDOF, non_zero_indices_list = matrixShrink(test_data, fix_indices_list) # Remove zero rows of data_x.
    data_x, mean_vect = zeroMean(data_x, training_ratio=1.0, mean_vect_input=mean_vect_list) # Shift(zero) the data to its mean (obtained from previous result). 
    data_y = np.transpose(eigVect) @ data_x
    
    test_dataloader, norm_params = dataProcessing(data_x, data_y, batch_size, FM_indices, bool_norm=isNormOn)

    pred_y_List, test_y_List, lossList_test = testNet(test_dataloader, neural_net, device)

    # Deformation reconstruction
    dist_nodal_matrix = np.zeros(shape=(int(test_data.shape[0]/3), len(pred_y_List)))
    test_reconstruct_list, mean_error_list, max_error_list = [], [], []

    for i in range(len(pred_y_List)):
        data_reconstruct = dataReconstruction(eigVect, eigVal, pred_y_List[i], mean_vect, 
                                              nDOF, non_zero_indices_list, bool_PC_norm=isPCNormOn) # Concatenated vector xyzxyz...; A transfer from training dataset (upon which the eigen-space is established) to testing dataset.
        dist_vector_temp = (data_reconstruct.reshape(-1,3) - 
                            test_data[:,i].reshape(-1,1).reshape(-1,3)) # Convert into node-wise matrix. 
        node_pair_distance = []

        for j in range(dist_vector_temp.shape[0]): # Number of nodes
            node_pair_distance.append(np.linalg.norm(dist_vector_temp[j,:]))
        
        mean_error_temp = np.sum(np.array(node_pair_distance).astype(float).reshape(-1,1)) / len(node_pair_distance)
        max_error_temp = np.max(node_pair_distance)
        dist_nodal_matrix[:,i] = np.array(node_pair_distance).astype(float).reshape(1,-1)
        test_reconstruct_list.append(data_reconstruct)
        mean_error_list.append(mean_error_temp)
        max_error_list.append(max_error_temp)
    
    # Pure PCA for test samples
    test_data_shrinked, _, _ = matrixShrink(test_data, fix_indices_list)
    if isPCNormOn: weights_test = np.transpose(eigVect/eigVal.reshape(-1,)) @ test_data_shrinked
    else: weights_test = np.transpose(eigVect) @ test_data_shrinked
    test_PCA_reconstruct = dataReconstruction(eigVect, eigVal, weights_test, mean_vect, 
                                              nDOF, non_zero_indices_list, bool_PC_norm=isPCNormOn)
    
    dist_nodal_matrix_testPCA = np.zeros(shape=(int(test_data.shape[0]/3), len(pred_y_List)))
    mean_error_list_testPCA, max_error_list_testPCA = [], []

    for i in range(test_PCA_reconstruct.shape[1]):
        dist_vector_temp = (test_PCA_reconstruct[:,i].reshape(-1,3) - 
                            test_data[:,i].reshape(-1,1).reshape(-1,3))
        node_pair_distance = []

        for j in range(dist_vector_temp.shape[0]): # Number of nodes
            node_pair_distance.append(np.linalg.norm(dist_vector_temp[j,:]))
        
        mean_error_temp = np.sum(np.array(node_pair_distance).astype(float).reshape(-1,1)) / len(node_pair_distance)
        max_error_temp = np.max(node_pair_distance)
        dist_nodal_matrix_testPCA[:,i] = np.array(node_pair_distance).astype(float).reshape(1,-1)
        mean_error_list_testPCA.append(mean_error_temp)
        max_error_list_testPCA.append(max_error_temp)

    max_nodal_error = 1e3*np.array(max_error_list).astype(float).reshape(-1,1) # Unit: mm. 
    mean_nodal_error = 1e3*np.array(mean_error_list).astype(float).reshape(-1,1) # Unit: mm. 
    max_mean = np.mean(max_nodal_error) # Compute the mean value of max errors. 
    mean_mean = np.mean(mean_nodal_error) # Compute the mean value of mean errors.

    # Save results to .mat files. 
    for i, vector in enumerate(test_reconstruct_list):
        if i == 0: test_reconstruct_matrix = vector
        else: test_reconstruct_matrix = np.concatenate((test_reconstruct_matrix, vector), axis=1)
    
    mdict = {"FM_num": FM_num, "PC_num": PC_num, "isPCNormOn": isPCNormOn, # Numbers of FMs and principal components. 
             "pred_y_list": pred_y_List, # The list of predicted weight vectors of test dataset.
             "test_deformation_label": test_data, # Label deformation results. 
             "test_deformation_reconstruct": test_reconstruct_matrix, # ANN reconstruction deformation results. 
             "test_PCA_reconstruct": test_PCA_reconstruct, # Reconstruction of pure PCA decomposition. 
             "fix_node_list": fix_indices_list, # List of fixed node indices. Indexed from 1. 
             "FM_indices": np.array(FM_indices).astype(int).reshape(-1,1) + 1, # FMs" indices. Add 1 to change to indexing system in Matlab. 
             "center_indices": np.array(center_indices_list).astype(int).reshape(-1,1) + 1, # Center indices generated from the k-center clustering. Add 1 to change to indexing system in Matlab. 
             "dist_nodal_matrix": 1e3*dist_nodal_matrix, # Distance between each nodal pair. Unit: mm
             "mean_nodal_error": mean_nodal_error, # Mean nodal distance of each sample. Unit: mm
             "max_nodal_error": max_nodal_error, # Max nodal distance of each sample. Unit: mm
             "eigVect_full": eigVect_full, "eigVal_full": eigVal_full, # Full eigenvector and eigenvalue matrices
             "eigVect": eigVect, "eigVal": eigVal, # Principal eigenvector and eigenvalue matrices
             "mean_vect": mean_vect, # The mean vector and principal eigenvector matrix of training dataset for data reconstruction. 
             "dist_nodal_matrix_testPCA": 1e3*dist_nodal_matrix_testPCA, # Distance between each nodal pair (pure PCA reconstruction). Unit: mm
             "mean_nodal_error_testPCA": 1e3*np.array(mean_error_list_testPCA).astype(float).reshape(-1,1), # Mean nodal distance of each sample (pure PCA reconstruction). Unit: mm
             "max_nodal_error_testPCA": 1e3*np.array(max_error_list_testPCA).astype(float).reshape(-1,1), # Max nodal distance of each sample (pure PCA reconstruction). Unit: mm
             "alpha_vector": alpha_vector # Vector of alphas of all tested samples. Size: sampleNum_test * 1. 
             }
    scipy.io.savemat("ANN_test_results.mat", mdict) # Run visualization on Matlab. 


if __name__ == "__main__":
    main()
