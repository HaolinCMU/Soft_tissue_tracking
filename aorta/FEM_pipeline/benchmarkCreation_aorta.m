close all; clear all; clc

%% load data

load data.mat
scale = 1e6;
maxDisp = 0.02; % max displacement
nMode = 20; % number of eigen force fields
nForceField = 1000; % number of eigen force fields.

[eigV, eigD] = eigs(laplacianMatrixI, nMode, 'sm'); % Shape's Laplacian -> shape's curvature info -> intrinsic. 

%% show eigen forces

nFigRow = 4;
nFigCol = 5;

figure
for i = 1 : size(eigV, 2)
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    color = zeros(nNodeI, 1);
    color(idxSurfI) = eigV(:, i);
    
    trisurf(FaceI, NodeI(:, 1), NodeI(:, 2), NodeI(:, 3), color, 'FaceColor', 'interp', 'EdgeColor', 'none');
    colormap jet
    axis equal off
    
end

%% create force field

weight = 2 * rand(nMode, 3 * nForceField) - 1; % x, y, z force fields. 
ForceField = eigV * weight * scale; % Force field reconstruction (not complete). 

% Set the force on inner surface to zero. 
for i = 1 : size(ForceField,1)
    for j = 1 : nFaceI
        temp = find(faces(j,2:end) == i);
        
        if size(temp, 2) == 0
            continue;
        else
            if faces(j, 1) ~= 1 % The region number of outer surface;
                ForceField(i,:) = 0;
            else
                continue;
            end
        end
    end
end


ForceField3 = zeros(3 * nSurfI, nForceField);
for i = 1 : nForceField
    ForceField3(:, i) = reshape(ForceField(:, 3 * i - 2 : 3 * i)', [], 1); % vector. 760*3 x 1. Concatenate to the order xyzxyz...
end

%% show example force fields (xyzxyz...)

nFigRow = 1;
nFigCol = 3;

figure
for i = 1 : 3
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    color = zeros(nNodeI, 1);
    color(idxSurfI) = ForceField(:, i);
    
    trisurf(FaceI, NodeI(:, 1), NodeI(:, 2), NodeI(:, 3), color, 'FaceColor', 'interp', 'EdgeColor', 'none');
    
    c = colorbar;
    c.Label.String = 'Pa';
    
    colormap jet
    caxis([min(min(ForceField(:, 1:3))), max(max(ForceField(:, 1:3)))]);
    axis equal off
    
end

%% show fixed nodes

idxFix = [117 429 1185]'; % FIx on two ends (nodes on regions 47 and 48). 

figure
hold on
plot3(NodeI(:, 1), NodeI(:, 2), NodeI(:, 3), 'b.');
camlight;
plot3(NodeI(idxFix,1), NodeI(idxFix,2), NodeI(idxFix,3),'r*');
axis equal

%% expand mass matrix and laplacian matrix to 3 dimension

massMatrixI3 = zeros(3 * nSurfI);
laplacianMatrixI3 = zeros(3 * nSurfI);
for i = 1 : 3  
    idxI3 = i : 3 : 3 * nSurfI;
    massMatrixI3(idxI3, idxI3) = massMatrixI;
    laplacianMatrixI3(idxI3, idxI3) = laplacianMatrixI;
end


%% create benchmark deformation using linear FEM

idxSurfI3 = reshape([idxSurfI * 3 - 2, idxSurfI * 3 - 1, idxSurfI * 3 - 0]', [], 1);
idxFix3 = reshape([idxFix * 3 - 2, idxFix * 3 - 1, idxFix * 3 - 0]', [], 1);
idxNonFix3 = 1 : size(KI, 1);
idxNonFix3(idxFix3) = [];
idxI = unique(FaceI(:));
idxI3 = reshape([idxI' * 3 - 2, idxI' * 3 - 1, idxI' * 3 - 0]', [], 1);

% force vector and stiffness matrix for solving
FIS = zeros(size(KI, 1), nForceField);
FIS(idxSurfI3, :) = massMatrixI3 * ForceField3; % Complete force field reconstruction. 
FIS(idxFix3, :) = [];

KIS = full(KI);
KIS(idxFix3, :) = [];
KIS(:, idxFix3) = [];

KIS = sparse(KIS);
XIS= KIS \ FIS; % Forward FEM. 

xI = zeros(size(KI, 1), nForceField);
xI(idxNonFix3, :) = XIS;

FI = KI * xI;

%% rescale the forcefield such that the max surface displacement equals to maxDisp

for i = 1:nForceField
    tX = reshape(xI(:, i), 3, [])';
    nodeDisp2 = sum(tX .* tX,  2);
    maxSurfDisp2 = max(nodeDisp2(idxSurfI, :));
    rescale = maxDisp / sqrt(maxSurfDisp2);
    xI(:, i) = xI(:, i) * rescale;
    FI(:, i) = FI(:, i) * rescale;
end

%% visualize benchmark deformations

nFigRow = 3;
nFigCol = 7;

figure
for i = 1 : nFigRow * nFigCol
    
    XIi = reshape(xI(:, i), 3, [])';
    
    left = (i - 1) / nFigCol - floor((i - 1) / nFigCol);
    bottom = 1 - (floor((i - 1) / nFigCol) + 1) / nFigRow;
    positionVector = [left, bottom, 1 / nFigCol, 1 / nFigRow];
    subplot('Position', positionVector);
    
    color = zeros(nNodeI, 1);
    color(idxSurfI) = ForceField(:, i);
    
    trisurf(FaceI, NodeI(:, 1) + XIi(:, 1), NodeI(:, 2) + XIi(:, 2), NodeI(:, 3) + XIi(:, 3), 'FaceColor', 'none', 'EdgeColor', [0 0.7 0]);
    hold on
    trisurf(FaceI, NodeI(:, 1), NodeI(:, 2), NodeI(:, 3),'FaceColor', 'none', 'EdgeColor', [0.25 0.25 1]);
    colormap jet
    axis equal off
    
end

deformation_matrix_block = zeros(size(KI, 1)/3, nForceField, 3);
for i=1:3
    matrix_temp = [];
    for j=i:3:size(KI, 1)
        matrix_temp = [matrix_temp;xI(j,:)];
    end
    deformation_matrix_block(:,:,i) = matrix_temp;
end

clear XIi
%%
benchmark_displacement = rand(15, 1000) .* 20.0;
weights = rand(20, 1000) - 0.5;
save benchmark.mat

