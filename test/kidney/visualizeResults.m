close all; clear all; clc

%% Load data & initialize plot parameters
load ANN_test_results.mat
load data_kidney.mat

nFigRow = 3;
nFigCol = 4;
starting_sample_NO = 1794; % 1794;

fix_node_list = [2, 453, 745];


%% Anchor nodes visualization in label configurations. 
figure
for i = 1 : nFigRow * nFigCol
    j = i + starting_sample_NO;
    
    deform_label = reshape(test_deformation_label(:, j), 3, [])'; % nNode * 3 matrix. 
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    trisurf(FaceI, NodeI(:, 1) + deform_label(:, 1), NodeI(:, 2) + deform_label(:, 2), NodeI(:, 3) + deform_label(:, 3),'FaceColor', 'none', 'EdgeColor', [0.25 0.25 1]); % Label configuration. 
    hold on
    plot3(NodeI(fix_node_list,1) + deform_label(fix_node_list, 1), NodeI(fix_node_list, 2) + deform_label(fix_node_list, 2), NodeI(fix_node_list, 3) + deform_label(fix_node_list, 3), 'r*'); % FMs in label configuration. 

    colormap jet
    axis equal off
    
end


%% FMs visualization in label configurations.
figure
for i = 1 : nFigRow * nFigCol
    j = i + starting_sample_NO;
    
    deform_label = reshape(test_deformation_label(:, j), 3, [])'; % nNode * 3 matrix. 
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    trisurf(FaceI, NodeI(:, 1) + deform_label(:, 1), NodeI(:, 2) + deform_label(:, 2), NodeI(:, 3) + deform_label(:, 3),'FaceColor', 'none', 'EdgeColor', [0.25 0.25 1]); % Label configuration. 
    hold on
    plot3(NodeI(FM_indices,1) + deform_label(FM_indices, 1), NodeI(FM_indices, 2) + deform_label(FM_indices, 2), NodeI(FM_indices, 3) + deform_label(FM_indices, 3), 'r*'); % FMs in label configuration. 

    colormap jet
    axis equal off
    
end


%% Centers visualization in label configurations.
figure
for i = 1 : nFigRow * nFigCol
    j = i + starting_sample_NO;
    
    deform_label = reshape(test_deformation_label(:, j), 3, [])'; % nNode * 3 matrix. 
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    trisurf(FaceI, NodeI(:, 1) + deform_label(:, 1), NodeI(:, 2) + deform_label(:, 2), NodeI(:, 3) + deform_label(:, 3),'FaceColor', 'none', 'EdgeColor', [0.25 0.25 1]); % Label configuration. 
    hold on
    plot3(NodeI(center_indices,1) + deform_label(center_indices, 1), NodeI(center_indices, 2) + deform_label(center_indices, 2), NodeI(center_indices, 3) + deform_label(center_indices, 3), 'r*'); % Center points in label configuration.

    colormap jet
    axis equal off
    
end


%% Label & ANN reconstruction & FMs
figure
for i = 1 : nFigRow * nFigCol
    j = i + starting_sample_NO;
    
    deform_label = reshape(test_deformation_label(:, j), 3, [])'; % nNode * 3 matrix. 
    deform_reconstruct = reshape(test_deformation_reconstruct(:, j), 3, [])'; % nNode * 3 matrix. 
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    trisurf(FaceI, NodeI(:, 1) + deform_reconstruct(:, 1), NodeI(:, 2) + deform_reconstruct(:, 2), NodeI(:, 3) + deform_reconstruct(:, 3), 'FaceColor', 'none', 'EdgeColor', [0 0.7 0]); % Reconstructed configuration from ANN. 
    hold on
    plot3(NodeI(FM_indices,1) + deform_reconstruct(FM_indices, 1), NodeI(FM_indices, 2) + deform_reconstruct(FM_indices, 2), NodeI(FM_indices, 3) + deform_reconstruct(FM_indices, 3), 'r*'); % FMs in ANN reconstructed configuration. 
    hold on
    trisurf(FaceI, NodeI(:, 1) + deform_label(:, 1), NodeI(:, 2) + deform_label(:, 2), NodeI(:, 3) + deform_label(:, 3),'FaceColor', 'none', 'EdgeColor', [0.25 0.25 1]); % Label configuration. 
    hold on
    plot3(NodeI(FM_indices,1) + deform_label(FM_indices, 1), NodeI(FM_indices, 2) + deform_label(FM_indices, 2), NodeI(FM_indices, 3) + deform_label(FM_indices, 3), 'yo'); % FMs in label configuration. 
    
    colormap jet
    axis equal off
    
end


%% Label & Pure PCA-based reconstruction & FMs
figure
for i = 1 : nFigRow * nFigCol
    j = i + starting_sample_NO;
    
    deform_label = reshape(test_deformation_label(:, j), 3, [])'; % nNode * 3 matrix. 
    test_PCA = reshape(test_PCA_reconstruct(:, j), 3, [])'; % nNode * 3 matrix. 
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    trisurf(FaceI, NodeI(:, 1) + deform_label(:, 1), NodeI(:, 2) + deform_label(:, 2), NodeI(:, 3) + deform_label(:, 3),'FaceColor', 'none', 'EdgeColor', [0.25 0.25 1]); % Label configuration. 
    hold on
    plot3(NodeI(FM_indices,1) + deform_label(FM_indices, 1), NodeI(FM_indices, 2) + deform_label(FM_indices, 2), NodeI(FM_indices, 3) + deform_label(FM_indices, 3), 'yo'); % FMs in label configuration. 
    hold on
    trisurf(FaceI, NodeI(:, 1) + test_PCA(:, 1), NodeI(:, 2) + test_PCA(:, 2), NodeI(:, 3) + test_PCA(:, 3),'FaceColor', 'none', 'EdgeColor', [1 0 0]); % Reconstructed configuration from PCA. 
    hold on
    plot3(NodeI(FM_indices,1) + test_PCA(FM_indices, 1), NodeI(FM_indices, 2) + test_PCA(FM_indices, 2), NodeI(FM_indices, 3) + test_PCA(FM_indices, 3), 'mo'); % FMs in PCA reconstructed configuration. 
    
    colormap jet
    axis equal off
    
end


%% All together
figure
for i = 1 : nFigRow * nFigCol
    j = i + starting_sample_NO;
    
    deform_label = reshape(test_deformation_label(:, j), 3, [])'; % nNode * 3 matrix. 
    deform_reconstruct = reshape(test_deformation_reconstruct(:, j), 3, [])'; % nNode * 3 matrix. 
    test_PCA = reshape(test_PCA_reconstruct(:, j), 3, [])'; % nNode * 3 matrix. 
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);

    trisurf(FaceI, NodeI(:, 1) + deform_reconstruct(:, 1), NodeI(:, 2) + deform_reconstruct(:, 2), NodeI(:, 3) + deform_reconstruct(:, 3), 'FaceColor', 'none', 'EdgeColor', [0 0.7 0]); % Reconstructed configuration from ANN. 
    hold on
    plot3(NodeI(FM_indices,1) + deform_reconstruct(FM_indices, 1), NodeI(FM_indices, 2) + deform_reconstruct(FM_indices, 2), NodeI(FM_indices, 3) + deform_reconstruct(FM_indices, 3), 'r*'); % FMs in ANN reconstructed configuration. 
    hold on
    trisurf(FaceI, NodeI(:, 1) + deform_label(:, 1), NodeI(:, 2) + deform_label(:, 2), NodeI(:, 3) + deform_label(:, 3),'FaceColor', 'none', 'EdgeColor', [0.25 0.25 1]); % Label configuration. 
    hold on
    plot3(NodeI(FM_indices,1) + deform_label(FM_indices, 1), NodeI(FM_indices, 2) + deform_label(FM_indices, 2), NodeI(FM_indices, 3) + deform_label(FM_indices, 3), 'yo'); % FMs in label configuration. 
    hold on
    trisurf(FaceI, NodeI(:, 1) + test_PCA(:, 1), NodeI(:, 2) + test_PCA(:, 2), NodeI(:, 3) + test_PCA(:, 3),'FaceColor', 'none', 'EdgeColor', [1 0 0]); % Reconstructed configuration from PCA. 
    hold on
    plot3(NodeI(FM_indices,1) + test_PCA(FM_indices, 1), NodeI(FM_indices, 2) + test_PCA(FM_indices, 2), NodeI(FM_indices, 3) + test_PCA(FM_indices, 3), 'mo'); % FMs in PCA reconstructed configuration. 
    
    colormap jet
    axis equal off
    
end
