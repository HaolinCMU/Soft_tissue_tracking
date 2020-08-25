close all; clear all; clc

%% Load data & initialize plot parameters
load ANN_benchmark_results.mat
load data.mat
load benchmark.mat

nFigRow = 3;
nFigCol = 4;

ForceField_test = ForceField(:, 901:end);


%% FMs visualization in label configurations.
figure
for i = 1 : nFigRow * nFigCol
    
    deform_label = reshape(test_deformation_label(:, i), 3, [])'; % nNode * 3 matrix. 
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    color = zeros(nNodeI, 1);
    color(idxSurfI) = ForceField(:, i);
    
    trisurf(FaceI, NodeI(:, 1) + deform_label(:, 1), NodeI(:, 2) + deform_label(:, 2), NodeI(:, 3) + deform_label(:, 3),'FaceColor', 'none', 'EdgeColor', [0.25 0.25 1]); % Label configuration. 
    hold on
    plot3(NodeI(FM_indices,1) + deform_label(FM_indices, 1), NodeI(FM_indices, 2) + deform_label(FM_indices, 2), NodeI(FM_indices, 3) + deform_label(FM_indices, 3), 'r*'); % FMs in label configuration. 
%     plot3(NodeI([42, 160, 493, 885, 1090],1) + deform_label([42, 160, 493, 885, 1090], 1), NodeI([42, 160, 493, 885, 1090], 2) + deform_label([42, 160, 493, 885, 1090], 2), NodeI([42, 160, 493, 885, 1090], 3) + deform_label([42, 160, 493, 885, 1090], 3), 'r*'); % FMs in label configuration. 
%     plot3(NodeI(center_indices,1) + deform_label(center_indices, 1), NodeI(center_indices, 2) + deform_label(center_indices, 2), NodeI(center_indices, 3) + deform_label(center_indices, 3), 'r*'); % Center points in label configuration.

    colormap jet
    axis equal off
    
end


%% Label & ANN reconstruction & FMs
figure
for i = 1 : nFigRow * nFigCol
    
    deform_label = reshape(test_deformation_label(:, i), 3, [])'; % nNode * 3 matrix. 
    deform_reconstruct = reshape(test_deformation_reconstruct(:, i), 3, [])'; % nNode * 3 matrix. 
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    color = zeros(nNodeI, 1);
    color(idxSurfI) = ForceField(:, i);
    
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
    
    deform_label = reshape(test_deformation_label(:, i), 3, [])'; % nNode * 3 matrix. 
    test_PCA = reshape(test_PCA_reconstruct(:, i), 3, [])'; % nNode * 3 matrix. 
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    color = zeros(nNodeI, 1);
    color(idxSurfI) = ForceField(:, i);
    
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
    
    deform_label = reshape(test_deformation_label(:, i), 3, [])'; % nNode * 3 matrix. 
    deform_reconstruct = reshape(test_deformation_reconstruct(:, i), 3, [])'; % nNode * 3 matrix. 
    test_PCA = reshape(test_PCA_reconstruct(:, i), 3, [])'; % nNode * 3 matrix. 
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    color = zeros(nNodeI, 1);
    color(idxSurfI) = ForceField(:, i);

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
