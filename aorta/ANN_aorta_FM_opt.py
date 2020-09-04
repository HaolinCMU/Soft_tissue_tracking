# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:11:26 2020

@author: haolinl
"""

import os
import copy

import numpy as np
import matplotlib.pyplot as plt
import scipy.io # For extracting data from .mat file
import scipy.stats as st
import time
import torch
import torch.utils.data
import torch.nn as nn


class Net1(nn.Module):
    """
    MLP modeling hyperparams:
    ----------
        Input: FM_num (FMs' displacements). 
        Hidden layers: Default architecture: 128 x 64. Optimization available. 
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
        # self.hidden_3 = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5)
        # )
        # self.hidden_4 = nn.Sequential(
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5)
        # )
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
        # f3 = self.hidden_3(f2)
        # f4 = self.hidden_4(f3)
        output = self.out_layer(f2)
        return output
        

def saveLog(lossList_train, lossList_valid, FM_num, PC_num, batch_size, learning_rate, 
            num_epochs, center_indices_list, elapsed_time, max_mean, mean_mean, write_path='train_valid_loss.log'):
    """
    Save the training & validation loss, training parameters and testing performance into .log file. 

    Parameters:
    ----------
        lossList_train: List. 
            The train loss of each epoch. Exact order.
        lossList_valid: List. 
            The valid loss of each epoch. Exact order.
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
        write_path (optional): String. 
            The path of to-be-saved .log file. 
            Default: 'train_valid_loss.log'
    """
    
    content = ['FM_num = {}'.format(FM_num),
               'FM_indices (indexed from 0) = {}'.format(list(np.sort(center_indices_list[0:FM_num]))),
               'Center_indices_list (indexed from 0) = {}'.format(list(center_indices_list)),
               'PC_num = {}'.format(PC_num), 
               'Batch_size = {}'.format(str(batch_size)), 
               'Learning_rate = {}'.format(str(learning_rate)),
               'Num_epochs = {}'.format(str(num_epochs)),
               '----------------------------------------------------------',
               'Epoch\tTraining loss\tValidation loss']
    
    for i in range(len(lossList_train)):
        loss_string_temp = '%d\t%.8f\t%.8f' % (i, lossList_train[i], lossList_valid[i])
        content.append(loss_string_temp)
    
    content += ['----------------------------------------------------------',
                'Elapsed_time = {} s'.format(elapsed_time),
                '\nTesting reconstruction performance parameters:',
                'Max_mean = %.8f mm' % (max_mean),
                'Mean_mean = %.8f mm' % (mean_mean)]
    content = '\n'.join(content)
    
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
            Containing 'min' and 'max'  of each direction for reconstruction. 
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


def matrixShrink(data_matrix):
    """
    Remove rows of zero displacement (fixed DOFs).

    Parameters:
    ----------
        data_matrix: 2D Array. 
            Size: nDOF x SampleNum. 
            The full matrix of deformation data.

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
    
    nDOF = data_matrix.shape[0]
    zero_indices_list, non_zero_indices_list = [], []
    for i in range(data_matrix.shape[0]):
        if data_matrix[i,0] == 0: zero_indices_list.append(i)
        else: non_zero_indices_list.append(i)
    
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
        data_new: 2D Array with the same size as data_matrix. Zeroshifted data. 
        mean_vect: 1D Array. 
            The mean value of each feature. 
    """
    
    mean_vect = np.mean(data_matrix, axis=1) # Compute mean along with sample's axis. 
    
    data_new = np.zeros(data_matrix.shape)
    for i in range(data_matrix.shape[1]):
        data_new[:,i] = data_matrix[:,i] - mean_vect
    
    return data_new, mean_vect


def PCA(data_matrix, PC_num):
    """
    Implement PCA on tumor's deformation data (Encoder). 

    Parameters:
    ----------
        data_matrix: 2D Array. 
            Size: nNodes*3 x SampleNum. 
            Each DOF is a feature. Zero shifted. 
        PC_num: Int. 
            The number of picked PCs.
    
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
    cov_matrix = data_matrix @ np.transpose(data_matrix) # Size: nDOF * nDOF
    eigVal_full, eigVect_full = np.linalg.eig(cov_matrix)
#    eigVect_full = data_matrix @ eigVect_full # Compute the eigenvectors of all features
    
    # PCA
    eigVal, eigVect = np.zeros(shape=(PC_num, 1), dtype=complex), np.zeros(shape=(eigVect_full.shape[0], PC_num), dtype=complex)
    eigVal_sorted_indices = np.argsort(np.real(eigVal_full))
    eigVal_PC_indices = eigVal_sorted_indices[-1:-(PC_num+1):-1] # Pick PC_num indices of largest principal eigenvalues
    
    for i, index in enumerate(eigVal_PC_indices): # From biggest to smallest
        eigVal[i,0] = eigVal_full[index] # Pick PC_num principal eigenvalues. Sorted. 
        eigVect[:,i] = eigVect_full[:,index] # Pick PC_num principal eigenvectors. Sorted. 
    
    # Compute weights of each sample on the picked basis (encoding). 
    weights = np.transpose(eigVect) @ data_matrix # Size: PC_num * SampleNum
    
    return eigVect_full, eigVal_full, eigVect, eigVal, weights


def dataReconstruction(eigVect, weights, mean_vect, nDOF, non_zero_indices_list):
    """
    Reconstruct the data with eigenvectors and weights (Decoder). 

    Parameters:
    ----------
        eigVect: 2D Array. 
            Size: nDOF x PC_num. 
            Principal eigenvectors aligned along with axis-1. 
        weights: 2D Array (complex). 
            Size: PC_num x SampleNum. 
            Weights of each sample aligned along with axis-1.
        mean_vect: 1D Array. 
            The mean value of each feature of training data. 
        nDOF: Int. 
            Number of all DOFs of original deformation matrix. 
        non_zero_indices_list: List. 
            All indices of non zero rows for deformation reconstruction. 
    
    Returns:
    ----------
        data_reconstruct: 2D Array. 
            Size: nDOF x SampleNum. 
            Reconstructed deformation results. 
    """
    
    # Transform weights back to original vector space (decoding)
    data_temp = eigVect @ weights # Complex matrix. 
    for i in range(data_temp.shape[1]):
        data_temp[:,i] += mean_vect # Shifting back
    
    data_reconstruct = np.zeros(shape=(nDOF, data_temp.shape[1]), dtype=complex)
    for i, index in enumerate(non_zero_indices_list):
        data_reconstruct[index,:] = data_temp[i,:]
    
    return np.real(data_reconstruct)
    

def performanceEvaluation():
    """
    Evaluate 3D Euclidean distance of each node pair of predicted and label deformation.
    """
    
    pass


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

            # ================= Picking only several points to calculate the distance variance (abandoned) ================= #
            # picked_num_temp = int(np.ceil(len(center_indices_list)*0.3)) # Pick several center points to compute the distance variance. 
            # picked_indices_temp = generateFMIndices(picked_num_temp, len(center_indices_list))
            # picked_indices_temp = [center_indices_list[i] for i in copy.deepcopy(picked_indices_temp)]
            # ============================================================================================================== #

            for i in range(v_space.shape[0]):
                if i in center_indices_list: continue

                coord_temp = v_space[i,:]
                dist_list = []

                # for index in picked_indices_temp: # Picking only several points to calculate the distance variance (abandoned). 
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


def generateFMIndices(FM_num, total_nodes_num):
    """
    Generate FM indices for benchmark deformation tracking. 

    Parameters:
    ----------
        FM_num: Int. 
            Number of FMs. 
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
        if rand_temp not in FM_indices: FM_indices.append(rand_temp)
    
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
            The label data (x SampleNum)
            Here it should be the weights vectors for the force field reconstruction. 
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
    
    # Partition the whole dataset into 'train' and 'test'. 
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


def trainValidateNet(train_dataloader, valid_dataloader, neural_net, learning_rate, 
                     num_epochs, neural_net_folderPath, iter_num, device):
    """
    Forward MLP training and validation. 

    Parameters:
    ----------
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
        iter_num: Int. 
            The number of the current iteration. 
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

        print("Iter: ", iter_num, "| Epoch: ", epoch, "| train loss: %.8f | valid loss: %.8f  " 
              % (loss_sum_train/iteration_num_train, loss_sum_valid/iteration_num_valid))
        
        if (epoch+1) % 100 == 0:
            ANN_savePath_temp = os.path.join(neural_net_folderPath, 
                                             'ANN_' + str(int((epoch+1)/100)) + '.pkl')
            torch.save(neural_net.state_dict(), ANN_savePath_temp) # Save the model every 100 epochs. 
    
    torch.save(neural_net.state_dict(), os.path.join(neural_net_folderPath, 'ANN_trained.pkl')) # Save the final trained ANN model.
    
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
    

if __name__ == "__main__":
    # Initialize hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 20
    learning_rate = 0.001
    num_epochs = 4000 # Default: 1500. 
    training_ratio = 0.8
    validation_ratio = 0.1
    FM_num = 5
    PC_num = 11 # Optimal. (Default: 11. From Dan 09/02). 
    isNormOn = False # True/Flase: Normalization on/off.
    ANN_folder_path = 'ANN_model' # The directory of trained ANN models. 
    figure_folder_path = 'figure' # The directory of figure folder. 
    isKCenter = True # True/Flase: Y/N for implementing optimized k-center. 

    if not os.path.isdir(ANN_folder_path): os.mkdir(ANN_folder_path)
    if not os.path.isdir(figure_folder_path): os.mkdir(figure_folder_path)


    # Extract data from .mat file
    data_mat = scipy.io.loadmat('benchmark.mat')
    v_space, data_x = data_mat['NodeI'], data_mat['xI'] # change the variable's name if necessary. 
    

    # DATA PROCESSING
    # Implement PCA
    data_x, nDOF, non_zero_indices_list = matrixShrink(data_x) # Remove zero rows of data_x.
    data_x, mean_vect = zeroMean(data_x) # Shift(zero) the data to its mean
    eigVect_full, eigVal_full, eigVect, eigVal, data_y = PCA(data_x, PC_num) # PCA on deformation matrix. 
    

    # Loop for picking optimal FMs. 
    iters, max_mean, mean_mean, opt_center_indices_list, tried_initial_pts_list = 0, 1e5, 1e5, None, []

    while(True):
        # Generate FM indices (Founded best initial indices: 217, 496, 523, 564, 584, 1063)
        if isKCenter:
            while(True):
                initial_pt_index = np.random.randint(0, int(data_x.shape[0] / 3.0)) # Initial point index for k-center clustering. Randomly assigned. Best result: 523 (best mean_max_nodal: 1.00 mm)
                if initial_pt_index not in tried_initial_pts_list:
                    tried_initial_pts_list.append(initial_pt_index)
                    break
            k = 20 # The number of wanted centers (must be larger than the FM_num). Default: 20. 
            style = "mean" # Style of k-center clustering. "mean" or "last". 
            center_indices_list = greedyClustering(v_space, initial_pt_index, k, style)
            if center_indices_list != []: FM_indices = center_indices_list[0:FM_num]
            else: 
                FM_indices = [584, 4, 268, 746, 303] # Optimal FM indices. Back-up choice when the returned list is empty.  
                center_indices_list = FM_indices 
        
        else:
            FM_indices = generateFMIndices(FM_num, int(data_x.shape[0] / 3.0)) # Randomly obtain FM indices. 
            # FM_indices = [876, 995, 1016, 867, 1034] # Manually choosing FMs. From k-means clustering. 
            # FM_indices = [584, 4, 268, 746, 303] # Manually choosing FMs. From k-center plus variation regularization. 
            # FM_indices = [42, 160, 493, 885, 1090] # Optimal FM indices. 
            center_indices_list = FM_indices
        

        # Generate train/valid/test tensor dataloaders. 
        (train_dataloader, valid_dataloader, 
        test_dataloader, norm_params) = dataProcessing(data_x, data_y,
                                                        batch_size, training_ratio, 
                                                        validation_ratio, FM_indices, 
                                                        bool_norm=isNormOn)
        

        # Generate MLP model
        neural_net = Net1(FM_num, PC_num).to(device)
        

        # Forward training & validation
        start_time = time.time()
        (neural_net, lossList_train, lossList_valid) = trainValidateNet(train_dataloader, valid_dataloader, 
                                                                        neural_net, learning_rate, num_epochs, ANN_folder_path, iters, device)
        end_time = time.time()
        elapsed_time = end_time - start_time # Elapsed time for training.      
        
        # Test MLP & Plot confidence interval of ANN accuracy
        pred_y_List, test_y_List, lossList_test = testNet(test_dataloader, neural_net, device)
        

        # Deformation reconstruction
        data_matrix = data_mat['xI']
        test_data = data_matrix[:,int(np.ceil(data_matrix.shape[1] * (training_ratio + validation_ratio))):] # Calling out testing deformation data
        dist_nodal_matrix = np.zeros(shape=(int(test_data.shape[0]/3), len(pred_y_List)))
        test_reconstruct_list, mean_error_list, max_error_list = [], [], []
        for i in range(len(pred_y_List)):
            data_reconstruct = dataReconstruction(eigVect, pred_y_List[i], mean_vect, 
                                                nDOF, non_zero_indices_list) # Concatenated vector xyzxyz...
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
        
        if np.mean(max_nodal_error) < max_mean: # Need a wise strategy to optimize FMs' distribution. 
            """
            Potential ideas: 
                1. Compute PCA on geometry's shape, then generate FMs based on the PCA base vectors. 
                2. Local vibration within Gaussian distribution. Need to know the the connectivity of each node. 
                3. Weighted navigation. Navigate the point along with the direction within which the testing performance is improved. 
                4. Derivative-free algorithm. Genetic algorithm / Simulated annealing. 
                5. K-center clustering / K-means clustering (adopted, with regularization of distance variance and constraint of minimum distance threshold. )
            """
            max_mean = np.mean(max_nodal_error)
            mean_mean = np.mean(mean_nodal_error)
            opt_center_indices_list = copy.deepcopy(center_indices_list)
            opt_FM_indices = np.sort(opt_center_indices_list[0:FM_num])
            torch.save(neural_net.state_dict(), 'ANN_optimal.pkl') # Save the optimal neural net. 
            # Save training process & test info into .log file.
            saveLog(lossList_train, lossList_valid, FM_num, PC_num, batch_size, 
                    learning_rate, num_epochs, center_indices_list, elapsed_time, max_mean, mean_mean)
        
        iters += 1
        # Break when the mean value of max euclidean distance is less than 1.0 mm, or reached maximum iter num. 
        # if max_mean < 1.0: break
        if iters > 200:
            print('Iterations reach the maximum number.')
            break

    print(opt_center_indices_list)
    print(opt_FM_indices)
