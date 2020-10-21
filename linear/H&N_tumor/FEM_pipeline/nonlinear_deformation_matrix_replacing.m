close all; clear all; clc

%% uhhh
load('benchmark20mm1000samples.mat');
load('C:\Users\13426\Desktop\soft_tissue_tracking\code\ANN\nonlinear\ANN_benchmark_results.mat', 'nonlinear_deformation_matrix');

xI = nonlinear_deformation_matrix(:,1:1000);


%% Save mat file. 
clear nonlinear_deformation_matrix
save('H&N_tumor_benchmark.mat');