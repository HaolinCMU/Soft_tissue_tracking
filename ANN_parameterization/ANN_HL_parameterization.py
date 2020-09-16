# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:11:26 2020

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
        Input: FM_num (FMs' displacements)
        Hidden layers: Default architecture: 128 x 64. Optimization available. 
        Output: PC_num (weights generated from deformation's PCA)
    """
    
    def __init__(self, FM_num, PC_num, hidden_layer_struct):
        """
        Parameters:
        ----------
            FM_num: Int. 
                The number of fiducial markers. 
            PC_num: Int. 
                The number of picked principal compoments. 
            hidden_layer_struct: List of int. 
                The architecture notation of hidden layers. 
                [a, b]: A hidden architecture with 2 layers; layer 1 has a neurons, layer 2 has b neurons. 
        """
        
        super(Net1, self).__init__()
        self.FM_num = FM_num
        self.PC_num = PC_num
        self.hidden_layer_struct = hidden_layer_struct
        self._hidden_layer_list = []
        self._network = None # The eventual neural network structure. 

        self.establishNetStruct() # Establish the ANN based on the hidden layer combination specified by "hidden_layer_struct". 
    

    def establishNetStruct(self):
        """
        Build the network architecture based on the `hidden_layer_struct`.

        Parameters (internal):
        ----------
            hidden_layer_struct: List of 1D lists. 
                The architecture notation of hidden layers. 
                [a, b]: A hidden architecture with 2 layers; layer 1 has a neurons, layer 2 has b neurons.

        Returns (internal):
        ----------
            _hidden_layer_list: List of linear layers. 
                Layers should be sorted in exact order from front to end. 
            _network: Entire MLP. 
                Concatenated and connected by implementing nn.Sequential() to hidden_layer_list. 
        """

        for index, neuron_num in enumerate(self.hidden_layer_struct):
            if index == 0:
                self._network = nn.Sequential(
                    nn.Linear(int(self.FM_num*3), neuron_num),
                    nn.ReLU(),
                    # nn.Dropout(0.5)
                )
                self._hidden_layer_list.append(copy.deepcopy(self._network))
            
            else:
                self._network = nn.Sequential(
                    nn.Linear(int(self.hidden_layer_struct[index-1]), neuron_num),
                    nn.ReLU(),
                    # nn.Dropout(0.5)
                )
                self._hidden_layer_list.append(copy.deepcopy(self._network))
            
        self._hidden_layer_list.append(nn.Linear(int(self.hidden_layer_struct[-1]), self.PC_num))
        self._network = nn.Sequential(*self._hidden_layer_list) # Concatenate all layers together for training. 

        
    def forward(self, x): # Something wrong here. Must be initialized, not storing in a list. 
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

        output = self._network(x)
        return output


def saveLog(lossList_train, lossList_valid, FM_num, PC_num, batch_size, learning_rate, 
            num_epochs, center_indices_list, elapsed_time, max_mean, mean_mean, write_path):
    """
    Save the training & validation loss, training parameters and testing performance into .log file. 

    Parameters:
    ----------
        lossList_train: List. 
            The train loss of each epoch. 
            In exact order.
        lossList_valid: List. 
            The valid loss of each epoch. 
            In exact order.
        FM_num: Int. 
            Number of fiducial markers. 
        PC_num: Int. 
            Number of principal components. 
        batch_size: Int. 
            The size of one single training batch. 
        learning_rate: Float. 
            Learning rate. 
        num_epochs: Int. 
            The number of total training iterations. 
        center_indices_list: List. 
            Picked indices of all generated centers/FMs. 
        elapsed_time: Float. 
            The time spent for training and validation process. 
            Unit: s. 
        max_mean: Float. 
            The mean value of max nodal errors of all test samples. 
            Unit: mm. 
        max_mean: Float. 
            The mean value of mean nodal errors of all test samples. 
            Unit: mm. 
        write_path: String. 
            The path of to-be-saved .log file. 
    """
    
    content = ["FM_num = {}".format(FM_num),
               "FM_indices (indexed from 0) = {}".format(list(np.sort(center_indices_list[0:FM_num]))),
               "Center_indices_list (indexed from 0, exact order) = {}".format(list(center_indices_list)),
               "PC_num = {}".format(PC_num), 
               "Batch_size = {}".format(str(batch_size)), 
               "Learning_rate = {}".format(str(learning_rate)),
               "Num_epochs = {}".format(str(num_epochs)),
               "----------------------------------------------------------",
               "Epoch\tTraining loss\tValidation loss"]
    
    for i in range(len(lossList_train)):
        loss_string_temp = "%d\t%.8f\t%.8f" % (i, lossList_train[i], lossList_valid[i])
        content.append(loss_string_temp)
    
    content += ["----------------------------------------------------------",
                "Elapsed_time = {} s".format(elapsed_time),
                "\nTesting reconstruction performance parameters:",
                "Max_mean = %.8f mm" % (max_mean),
                "Mean_mean = %.8f mm" % (mean_mean)]
    content = '\n'.join(content)
    
    with open(write_path, 'w') as f: f.write(content)


def saveParameterizationLog(results_list, write_path, FM_num, PC_num, batch_size, learning_rate, 
                            num_epochs, center_indices_list, training_ratio, validation_ratio, repeat_training_iters):
    """
    Save the parameterization results in a .log file. 

    Parameters:
    ----------
        results_list: List of string. 
            Record the training&validation time and euclidean error of reconstruction.
        write_path: String. 
            The path to save the .log file. 
        FM_num: Int. 
            Number of fiducial markers. 
        PC_num: Int. 
            Number of principal components. 
        batch_size: Int. 
            The size of one single training batch. 
        learning_rate: Float. 
            Learning rate. 
        num_epochs: Int. 
            The number of total training iterations. 
        center_indices_list: List. 
            Picked indices of all generated centers/FMs. 
        training_ratio: Float. 
            The portion of training dataset. 
        validation_ratio: Float. 
            The portion of validation dataset. 
        repeat_training_iters: Int. 
            THe number of times the training repeats for each fixed model. 
    """

    header_list = ["FM_num = {}".format(FM_num),
                   "FM_indices (indexed from 0) = {}".format(list(np.sort(center_indices_list[0:FM_num]))),
                   "Center_indices_list (indexed from 0, exact order) = {}".format(list(center_indices_list)),
                   "PC_num = {}".format(PC_num), 
                   "Batch_size = {}".format(str(batch_size)), 
                   "Learning_rate = {}".format(str(learning_rate)),
                   "Num_epochs = {}".format(str(num_epochs)),
                   "Training_ratio = {}".format(str(training_ratio)),
                   "Validation_ratio = {}".format(str(validation_ratio)),
                   "Repeat_training_iters = {}".format(str(repeat_training_iters)),
                   "----------------------------------------------------------"]
    header_list += results_list

    content = '\n'.join(header_list)
    with open(write_path, 'w') as f: f.write(content)


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
            Row order: [x_min;x_max;y_min;y_max;z_min;z_max]. 
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
        dix_indices_list (optional): List of ints.
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
    

def zeroMean(data_matrix):
    """
    Shift the origin of new basis coordinate system to mean point of the data. 

    Parameters:
    ----------
        data_matrix: 2D Array. 
            Size: nFeatures x nSamples. 

    Returns:
    ----------
        data_new: 2D Array with the same size as data_matrix. 
            Mean-shifted data. 
        mean_vect: 1D Array. 
            The mean value of each feature. 
    """
    
    mean_vect = np.mean(data_matrix, axis=1) # Compute mean along with sample's axis. 
    data_new = np.zeros(data_matrix.shape)

    for i in range(data_matrix.shape[1]):
        data_new[:,i] = data_matrix[:,i] - mean_vect
    
    return data_new, mean_vect


def PCA(data_matrix, PC_num, training_ratio):
    """
    Implement PCA on tumor's deformation covariance matrix (Encoder). 

    Parameters:
    ----------
        data_matrix: 2D Array. 
            Size: nNodes*3 x SampleNum. 
            Each DOF is a feature. Mean-shifted.  
        PC_num: Int. 
            The number of picked PCs.
        training_ratio: float.
            The ratio of training dataset.

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
    
    # PCA
    eigVal, eigVect = np.zeros(shape=(PC_num, 1), dtype=complex), np.zeros(shape=(eigVect_full.shape[0], PC_num), dtype=complex)
    eigVal_sorted_indices = np.argsort(np.real(eigVal_full))
    eigVal_PC_indices = eigVal_sorted_indices[-1:-(PC_num+1):-1] # Pick PC_num indices of largest principal eigenvalues
    
    for i, index in enumerate(eigVal_PC_indices): # From biggest to smallest
        eigVal[i,0] = eigVal_full[index] # Pick PC_num principal eigenvalues. Sorted. 
        eigVect[:,i] = eigVect_full[:,index] # Pick PC_num principal eigenvectors. Sorted. 
    
    # Compute weights of each sample on the picked basis (encoding). 
    weights = np.transpose(eigVect) @ data_matrix # Size: PC_num * SampleNum, complex. 
    
    return eigVect_full, eigVal_full, eigVect, eigVal, weights


def dataReconstruction(eigVect, weights, mean_vect, nDOF, non_zero_indices_list):
    """
    Reconstruct the data with eigenvectors and weights (Decoder). 

    Parameters:
    ----------
        eigVect: 2D Array. 
            Principal eigenvectors aligned along with axis-1. 
            Size: nDOF x PC_num. 
        weights: 2D Array (complex). 
            Weights of each sample aligned along with axis-1. 
            Size: PC_num x SampleNum. 
        mean_vect: 1D Array. 
            The mean value of each feature of training data. 
        nDOF: Int. 
            Number of all DOFs of original deformation matrix. 
        non_zero_indices_list: List. 
            All indices of non zero rows for deformation reconstruction. 

    Returns:
    ----------
        data_reconstruct: 2D Array. 
            Reconstructed deformation results. 
            Size: nDOF x SampleNum. 
    """
    
    # Transform weights back to original vector space (decoding)
    data_temp = eigVect @ weights

    for i in range(data_temp.shape[1]):
        data_temp[:,i] += mean_vect # Shifting back
    
    data_reconstruct = np.zeros(shape=(nDOF, data_temp.shape[1]), dtype=complex)

    for i, index in enumerate(non_zero_indices_list):
        data_reconstruct[index,:] = data_temp[i,:]
    
    return np.real(data_reconstruct)
    

def greedyClustering(v_space, initial_pt_index, k, style):
    """
    Generate `k` centers, starting with the `initial_pt_index`.

    Parameters: 
    ----------
        v_space: 2D array. 
            The coordinate matrix of the initial geometry. 
            The column number is the vertex's index. 
        initial_pt_index: Int. 
            The index of the initial point. 
        k: Int. 
            The number of centers aiming to generate. 
        style: String. 
            Indicate "last" or "mean" to choose the style of evaluation function. 
                "last": Calculate the farthest point by tracking the last generated center point. 
                        Minimum distance threshold applied. 
                "mean": Calculate a point with the maximum average distance to all generated centers; 
                        Calculate a point with the minimum distance variance of all generated centers; 
                        Minimum distance threshold applied. 

    Returns:
    ----------
        center_indices_list: List of int. 
            Containing the indices of all k centers. 
            Empty if the input style indicator is wrong. 
    """

    if style == "last":
        center_indices_list = []
        center_indices_list.append(initial_pt_index)
        min_dist_thrshld = 0.01 # Unit: m. The radius of FM ball. 

        for j in range(k):
            center_coord_temp = v_space[center_indices_list[j],:]
            max_dist_temp = 0.0
            new_center_index_temp = 0

            for i in range(v_space.shape[0]):
                if i in center_indices_list: continue
            
                coord_temp = v_space[i,:]
                dist_temp = np.linalg.norm(center_coord_temp.reshape(-1,3) - coord_temp.reshape(-1,3))
                dist_list = []
                
                for index in center_indices_list:
                    dist_temp_eachCenter = np.linalg.norm(coord_temp.reshape(-1,3) - v_space[index,:].reshape(-1,3))
                    dist_list.append(dist_temp_eachCenter)
                
                min_dist_temp = np.min(dist_list)

                if dist_temp > max_dist_temp and min_dist_temp >= min_dist_thrshld: 
                    max_dist_temp = dist_temp
                    new_center_index_temp = i
            
            if new_center_index_temp not in center_indices_list:
                center_indices_list.append(new_center_index_temp)
        
        return center_indices_list
    
    elif style == "mean":
        center_indices_list = []
        center_indices_list.append(initial_pt_index)
        min_dist_thrshld = 0.01 # Unit: m. The radius of FM ball. 

        while(True):
            max_dist_thrshld = 0.0
            new_center_index_temp = 0

            for i in range(v_space.shape[0]):
                if i in center_indices_list: continue
            
                coord_temp = v_space[i,:]
                dist_list = []

                for index in center_indices_list:
                    dist_temp = np.linalg.norm(coord_temp.reshape(-1,3) - v_space[index,:].reshape(-1,3))
                    dist_list.append(dist_temp)

                avg_dist_temp = np.mean(dist_list)
                min_dist_temp = np.min(dist_list)

                if avg_dist_temp > max_dist_thrshld and min_dist_temp >= min_dist_thrshld: 
                    max_dist_thrshld = avg_dist_temp
                    new_center_index_temp = i
            
            if new_center_index_temp not in center_indices_list:
                center_indices_list.append(new_center_index_temp)

            if len(center_indices_list) >= k: break

            var_thrshld = 1e5
            new_center_index_temp = 0

            for i in range(v_space.shape[0]):
                if i in center_indices_list: continue

                coord_temp = v_space[i,:]
                dist_list = []
 
                for index in center_indices_list:
                    dist_temp = np.linalg.norm(coord_temp.reshape(-1,3) - v_space[index,:].reshape(-1,3))
                    dist_list.append(dist_temp)

                var_dist_temp = np.var(dist_list)
                min_dist_temp = np.min(dist_list)

                if var_dist_temp < var_thrshld and min_dist_temp >= min_dist_thrshld: 
                    var_thrshld = var_dist_temp
                    new_center_index_temp = i
            
            if new_center_index_temp not in center_indices_list:
                center_indices_list.append(new_center_index_temp)

            if len(center_indices_list) >= k: break
        
        return center_indices_list
    
    else: 
        print("Wrong input of the style indicator. Will start training based on the optimal FM indices. ")
        return []


def generateFMIndices(FM_num, fix_node_list, total_nodes_num):
    """
    Generate FM indices for benchmark deformation tracking. 

    Parameters:
    ----------
        FM_num: Int. 
            Number of FMs. 
        fix_node_list: List of ints. 
            Indices of fixed nodes. 
        total_nodes_num: Int. 
            Total number of nodes. 
    
    Returns:
    ----------
        FM_indices: List of int. 
            Random ints (indices) within the range of [0, total_nodes_num]. 
    """

    FM_indices = []

    for i in range(FM_num):
        rand_temp = np.random.randint(0, total_nodes_num)
        if (rand_temp not in FM_indices and 
            rand_temp+1 not in fix_node_list): FM_indices.append(rand_temp)
    
    return FM_indices


def dataProcessing(data_x, data_y, batch_size, training_ratio, validation_ratio,
                   FM_indices, bool_norm=False):
    """
    Data preprocessing. 

    Parameters:
    ----------
        data_x: 2D Array (nDOF x SampleNum). 
            The deformation data (x SampleNum) of all DOFs. 
        data_y: 2D Array (PC_num x SampleNum, complex). 
            The label data (x SampleNum), here it should be the weights vectors for the force field reconstruction. 
        batch_size: Int. 
            The size of a single training batch input.
        training_ratio: Float. 
            Indicates the portion of training dataset. 
        validation_ratio: Float. 
            Indicates the portion of validation dataset. 
        FM_indices: 1D Array. 
            Randomly picked FM indices. 
            Typical size: 5. 
        bool_norm (optional): Boolean. 
            True: conduct directional input normalization. 
            False: skip directional input normalization. 
            Default: False. 

    Returns:
    ----------
        train_dataloader: Tensor dataloader. 
            Training dataset.
        valid_dataloader: Tensor dataloader. 
            Validation dataset.
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
    training_index = int(np.ceil(data_x.shape[1] * training_ratio)) # Samples along with axis-1.
    validation_index = int(np.ceil(data_x.shape[1] * (training_ratio + validation_ratio))) # Samples along with axis-1.
    train_x = torch.from_numpy(data_x[:,0:training_index]).float() # size: 15 x nTrain
    train_y = torch.from_numpy(data_y[:,0:training_index]).float() # size: 20 x nTrain
    valid_x = torch.from_numpy(data_x[:,training_index:validation_index]).float() # size: 15 x nValid
    valid_y = torch.from_numpy(data_y[:,training_index:validation_index]).float() # size: 20 x nValid
    test_x = torch.from_numpy(data_x[:,validation_index:]).float() # size: 15 x nTest
    test_y = torch.from_numpy(data_y[:,validation_index:]).float() # size: 20 x nTest
    
    # Generate dataloaders 
    # Make sure the sample dimension is on axis-0. 
    train_dataset = torch.utils.data.TensorDataset(np.transpose(train_x), 
                                                   np.transpose(train_y))
    valid_dataset = torch.utils.data.TensorDataset(np.transpose(valid_x), 
                                                   np.transpose(valid_y))
    test_dataset = torch.utils.data.TensorDataset(np.transpose(test_x), 
                                                  np.transpose(test_y))
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset
    )
    
    return train_dataloader, valid_dataloader, test_dataloader, norm_params


def trainValidateNet(combination, iter_num, train_dataloader, valid_dataloader, neural_net, learning_rate, 
                     num_epochs, neural_net_folderPath, device):
    """
    Forward MLP training and validation. 

    Parameters:
    ----------
        combination: List of int. 
            The neuron number combination. 
        iter_num: Int. 
            The number of (repeating) iteration the model is being trained. 
        train_dataloader: Tensor dataloader. 
            Training dataset.
        valid_dataloader: Tensor dataloader. 
            Validation dataset.
        neural_net: MLP model.
        learning_rate: Float. 
            Specify a value typically less than 1.
        num_epochs: Int. 
            Total number of training epochs. 
        neural_net_folderPath: String. 
            The directory to save the eventual trained ANN. 
        device: CPU/GPU. 

    Returns:
    ----------
        neural_net: Trained MLP. 
        lossList_train: List. 
            The loss result of each training epoch.
        lossList_valid: List. 
            The loss result of each validation epoch. 
    """
    
    # Define criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), learning_rate)
    
    # Iterative training and validation
    lossList_train, lossList_valid = [], [] # List of loss summation during training and validation process.  
    
    for epoch in range(num_epochs):
        loss_sum_train, loss_sum_valid = 0, 0
        
        # Training
        iteration_num_train = 0
        for iteration, (displacements, weights) in enumerate(train_dataloader):
            # Forward fitting
            x_train_batch = torch.autograd.Variable(displacements)
            y_train_batch = torch.autograd.Variable(weights)
            x_train_batch = x_train_batch.to(device)
            y_train_batch = y_train_batch.to(device)
            output = neural_net(x_train_batch)
            loss_train_temp = criterion(output, y_train_batch)
            
            # Back propagation
            optimizer.zero_grad()
            loss_train_temp.backward()
            optimizer.step()

            loss_sum_train += loss_train_temp.cpu().data.numpy()
            iteration_num_train += 1

        lossList_train.append(loss_sum_train/iteration_num_train)
        
        # Validation
        iteration_num_valid = 0
        for iteration, (displacements, weights) in enumerate(valid_dataloader):
            x_valid_batch = torch.autograd.Variable(displacements)
            y_valid_batch = torch.autograd.Variable(weights)
            x_valid_batch = x_valid_batch.to(device)
            y_valid_batch = y_valid_batch.to(device)
            output = neural_net(x_valid_batch)
            loss_valid_temp = criterion(output, y_valid_batch)

            loss_sum_valid += loss_valid_temp.cpu().data.numpy()
            iteration_num_valid += 1
        
        lossList_valid.append(loss_sum_valid/iteration_num_valid)

        print("Archi: {} | Iter: ".format(combination), iter_num+1, "| Epoch: ", epoch, "| train loss: %.8f | valid loss: %.8f  " 
              % (loss_sum_train/iteration_num_train, loss_sum_valid/iteration_num_valid))
    
    name_label_list_temp = [str(item) for item in neural_net.hidden_layer_struct]
    name_label_string = '_'.join(name_label_list_temp)
    torch.save(neural_net.state_dict(), os.path.join(neural_net_folderPath, "ANN_trained_{}.pkl".format(name_label_string))) # Save the trained ANN model from the last training iter.
    
    return neural_net, lossList_train, lossList_valid


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


def combinationRecursion(neuron_num_options_list, layer_num, combination_list_temp, combination_list_total):
    """
    Recursively generate all valid combinations of hidden layers. 

    Parameters:
    ----------
        neuron_num_options_list: List of int. 
            List all possible neuron numbers for a single hidden layer. 
        layer_num: Int. 
            The number of intended hidden layers. 
        combination_list_temp: List of int. 
            A single combination of a hidden layer. Appendable. 
        combination_list_total: Multi-dimentional list of int. 
            Store all possible combinations recursively generated by the function.

    Returns:
    ----------
        combination_list_temp: List of int. 
            Recursive return. 
            Return the updated combination during recursion process. 
        combination_list_total: Multi-dimentional list of int. 
            Store all possible combinations recursively generated by the function.
    """

    for item in neuron_num_options_list:
        combination_list_temp.append(item)
        
        if len(combination_list_temp) >= layer_num:
            combination_list_total.append(copy.deepcopy(combination_list_temp))
            combination_list_temp = copy.deepcopy(combination_list_temp[:-1])
            
        else: 
            combination_list_temp, combination_list_total = combinationRecursion(neuron_num_options_list, layer_num, combination_list_temp, combination_list_total)
            combination_list_temp = copy.deepcopy(combination_list_temp[:-1])
    
    return combination_list_temp, combination_list_total


def buildHiddenLayerStructs(neuron_num_options_list, layer_num_range, add_cases=[], skip_cases=[]):
    """
    Generate a 2D list containing all combinations of hidden layer architectures. 

    Parameters:
    ----------
        neuron_num_options_list: List of int. 
            List all possible neuron numbers for a single hidden layer. 
        layer_num_range: Tuple of int. 
            (minimum_layer_num, maximum_layer_num). Closed tuple. 
            Indicate the range of layer numbers, within which all possible combinations will be tested. 
        add_cases (optional): 2D list of int. 
            Specify the combinations you want to add. 
            Default: []. 
        skip_cases (optional): 2D list of int. 
            Specify the combinations you want to skip. 
            Default: []. 

    Returns:
    ----------
        combination_list_total: Multi-dimentional list of int. 
            Store all possible combinations recursively generated by the function.
    """

    combination_list_total = []

    for num_temp in range(layer_num_range[0], layer_num_range[1]+1):
        combination_list_temp = []
        _, combination_list_total = combinationRecursion(neuron_num_options_list, num_temp, combination_list_temp, combination_list_total)
    
    for list_temp in add_cases: combination_list_total.append(list_temp) # Add combinations from "add_cases". 

    for list_temp in combination_list_total:
        if list_temp in skip_cases: combination_list_total.remove(list_temp) # Remove combinations from "skip_cases". 
    
    return copy.deepcopy(combination_list_total)


def main(): 
    """
    MAIN IMPLEMENTATION AND EXECUTION. 

    Preparations:
    ----------
        1. Run benchmarkCreation.m in Matlab to generate the file "benchmark20mm1000samples.mat" (main data file) in the working directory;
        2. Create an empty folders in the working directory and name it as "ANN_parameterization".

    Pipeline:
    ----------
        1. Initialize parameters;
        2. Extract data from the aforementioned .mat files;
        3. Implement PCA on the extracted data, and generate/obtain the fiducial marker indices;
        4. Data preprocessing, and generate train/valid/test tensor dataloaders; 
        5. Iteratively Train & Validate & Test ANN with different MLP architectures (main difference: combination/structure of hidden layers);
        6. Deformation reconstruction for ANN; 
        7. Save logs for each model and the overall parameterization.

    Result files: 
    ----------
        1. "ANN_trained_*.pkl" x combination_num. 
            The model/parameter files of trained ANN with different hidden layer architectures. 
            Automatically saved in the folder "ANN_parameterization"; 
        2. "train_valid_loss_*.log". 
            The text file contains hyperparameters of each ANN, loss & elapsed time of the training-validation process, and the performance of each model's testing; 
        3. "parameterization_results.log". 
            Mean values of max nodal errors and elapsed time results of all tested models.  

    Next steps: 
    ----------
        1. Change the hidden layer architecture of "ANN_64x32.py" to the optimal one from the parameterization, then conduct the pipeline of "ANN_64x32.py". 
        2. Run the file "ANN_64x32_FM_opt.py" with the optimal hidden layer architecture again, and find the optimal initlal FM and the corresponding center point indices in a certain distributed order. 
    """

    # ********************************** INITIALIZE PARAMETERS ********************************** #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 20
    learning_rate = 0.001
    num_epochs = 4000 # Default: 1500. 
    training_ratio = 0.8
    validation_ratio = 0.1
    FM_num = 5
    PC_num = 27 # Optimal. (Default: 27. From Dan 09/02). 
    isNormOn = False # True/Flase: Normalization on/off.
    ANN_folder_path = "ANN_parameterization" # The directory of trained ANN models. 
    figure_folder_path = "figure" # The directory of figure folder. 
    isKCenter = True # True/Flase: Y/N for implementing optimized k-center. 

    if not os.path.isdir(ANN_folder_path): os.mkdir(ANN_folder_path)
    if not os.path.isdir(figure_folder_path): os.mkdir(figure_folder_path)
    
    # Parameterization related parameters. 
    # Define ANN hidden layer combinations. 
    neuron_num_options_list = [32, 64, 128] # List of int. List all possible neuron numbers for a single hidden layer. 
    layer_num_range = (2, 2) # Tuple of int. (minimum_layer_num, maximum_layer_num). Indicate the range of layer numbers, within which all possible combinations will be tested. Closed tuple. 
    abandoned_combinations = [] # 2D list of int. Specify the abandoned combinations.
    additional_combinations = [[32, 32, 32],
                               [64, 64, 64],
                               [128, 128, 128],
                               [128, 64, 32],
                               [32, 128, 64],
                               [64, 128, 32],
                               [64, 32, 32],
                               [128, 64, 64]] # 2D list of int. Specify the additional combinations.
    hidden_layer_architectures = buildHiddenLayerStructs(neuron_num_options_list, layer_num_range, 
                                                         add_cases=additional_combinations, skip_cases=abandoned_combinations) # List containing all hidden layer combinations. 

    # Specify the time for repeat training. 
    repeat_training_iters = 3


    # ********************************** DATA PROCESSING ********************************** #
    # Extract data from .mat file
    data_mat = scipy.io.loadmat("benchmark20mm1000samples.mat")
    v_space, data_x, fix_dof_list = data_mat["NodeI"], data_mat["xI"], data_mat["idxFix3"] # change the variable's name if necessary. 
    fix_node_list = [ind for ind in fix_dof_list if ind % 3 == 0] # Indices of fixed nodes. Indexed from 1. 

    # Implement PCA
    orig_node_num = int(data_x.shape[0] / 3.0)
    data_x, nDOF, non_zero_indices_list = matrixShrink(data_x) # Remove zero rows of data_x.
    data_x, mean_vect = zeroMean(data_x) # Shift(zero) the data to its mean
    eigVect_full, eigVal_full, eigVect, eigVal, data_y = PCA(data_x, PC_num, training_ratio) # PCA on training deformation matrix. 

    # Generate FM indices
    v_space, _, _ = matrixShrink(v_space, fix_node_list)
    if isKCenter:
        initial_pt_index = 96 # Initial point index for k-center clustering. Randomly assigned. Current best result: 584 (best mean_max_nodal_error: 0.92 mm)
        k = 20 # The number of wanted centers (must be larger than the FM_num). Default: 20. 
        style = "mean" # Style of k-center clustering. "mean" or "last". 
        center_indices_list = greedyClustering(v_space, initial_pt_index, k, style)
        if center_indices_list != []: FM_indices = center_indices_list[0:FM_num]
        else: 
            FM_indices = [4, 96, 431, 752, 1144] # Optimal FM indices. Back-up choice when the returned list is empty.  
            center_indices_list = FM_indices
    
    else:
        FM_indices = generateFMIndices(FM_num, fix_node_list, orig_node_num) # Randomly obtain FM indices. 
        center_indices_list = FM_indices

    # Generate train/valid/test tensor dataloaders. 
    (train_dataloader, valid_dataloader, 
     test_dataloader, norm_params) = dataProcessing(data_x, data_y,
                                                    batch_size, training_ratio, 
                                                    validation_ratio, FM_indices, 
                                                    bool_norm=isNormOn)
    

    # ********************************** TRAIN/VALID/TEST & PERFORMANCE EVALUATION ********************************** #
    results_list = ["Combination\t\tmean_max_nodal_error/mm\t\tmean_mean_nodal_error/mm\t\telapsed_time/s"]
    
    for hidden_layer_struct in hidden_layer_architectures: # List. Extract a single combination of hidden layers in each iteration. 
        mean_max_err_list, mean_mean_err_list, elapsed_time_list = [], [], []

        for i in range(repeat_training_iters):
            # Generate MLP model
            neural_net = Net1(FM_num, PC_num, hidden_layer_struct).to(device)
            
            # Forward training & validation
            start_time = time.time()
            neural_net, lossList_train, lossList_valid = trainValidateNet(hidden_layer_struct, i, train_dataloader, valid_dataloader, 
                                                                          neural_net, learning_rate, num_epochs, ANN_folder_path, device)
            end_time = time.time()
            elapsed_time = end_time - start_time # Elapsed time for training. 

            # Test pre-trained MLP & Plot confidence interval of ANN accuracy
            pred_y_List, test_y_List, lossList_test = testNet(test_dataloader, neural_net, device)

            # Deformation reconstruction
            data_matrix = data_mat["xI"]
            test_data = data_matrix[:,int(np.ceil(data_matrix.shape[1] * (training_ratio + validation_ratio))):] # Calling out testing deformation data
            dist_nodal_matrix = np.zeros(shape=(int(test_data.shape[0]/3), len(pred_y_List)))
            test_reconstruct_list, mean_error_list, max_error_list = [], [], []

            for i in range(len(pred_y_List)):
                data_reconstruct = dataReconstruction(eigVect, pred_y_List[i], mean_vect, 
                                                    nDOF, non_zero_indices_list) # Concatenated vector xyzxyz...; A transfer from training dataset (upon which the eigen-space is established) to testing dataset. 
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
            
            max_nodal_error = 1e3*np.array(max_error_list).astype(float).reshape(-1,1) # Unit: mm. 
            mean_nodal_error = 1e3*np.array(mean_error_list).astype(float).reshape(-1,1) # Unit: mm.
            max_mean = np.mean(max_nodal_error) # Compute the mean value of max errors. 
            mean_mean = np.mean(mean_nodal_error) # Compute the mean value of mean errors.

            mean_max_err_list.append(max_mean)
            mean_mean_err_list.append(mean_mean)
            elapsed_time_list.append(elapsed_time)

        # Save training process & test info & parameterization results into .log files. 
        name_label_list_temp = [str(item) for item in neural_net.hidden_layer_struct]
        name_label_string = '_'.join(name_label_list_temp)
        writePath_ANN_models = os.path.join(ANN_folder_path, "train_valid_loss_{}.log".format(name_label_string))

        saveLog(lossList_train, lossList_valid, FM_num, PC_num, batch_size, 
                learning_rate, num_epochs, center_indices_list, elapsed_time, max_mean, mean_mean, write_path=writePath_ANN_models) # Save the training results from the last training iter.
        
        max_mean_mean, mean_mean_mean, elapsed_time_mean = np.mean(mean_max_err_list), np.mean(mean_mean_err_list), np.mean(elapsed_time_list)
        print_string_temp = ("{}".format(neural_net.hidden_layer_struct) + 
                             "\t\t\t%.4f\t\t\t\t%.4f\t\t\t\t%.4f" % (max_mean_mean, mean_mean_mean, elapsed_time_mean))
        results_list.append(print_string_temp)
        writePath_parameterization_results = os.path.join(ANN_folder_path, "parameterization_results.log")

        saveParameterizationLog(results_list, writePath_parameterization_results, FM_num, PC_num, batch_size, learning_rate, 
                                num_epochs, center_indices_list, training_ratio, validation_ratio, repeat_training_iters)


if __name__ == "__main__":
    # Run the main function in terminal: python ANN_parameterization.py
    main()