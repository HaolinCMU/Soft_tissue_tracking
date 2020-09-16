close all; clear all; clc

%% Force field visualization. 

load("force_field.mat");
load("benchmark.mat");

nFigRow = 1;
nFigCol = 3;

figure
for i = 1 : 3
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    color = zeros(nNodeI, 1);
    color(idxSurfI) = force_field(:, i);
    
    trisurf(FaceI, NodeI(:, 1), NodeI(:, 2), NodeI(:, 3), color, 'FaceColor', 'interp', 'EdgeColor', 'none');
    
    c = colorbar;
    c.Label.String = 'Pa';
    
    colormap jet
    
    caxis([min(min(force_field(:, 1:3))), max(max(force_field(:, 1:3)))]);
    axis equal off
    
end
