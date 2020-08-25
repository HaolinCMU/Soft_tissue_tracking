close all; clear all; clc

%% inputs
% read file core mesh
file_path = 'kidney_volume_cortex';
[NodeI, faces, elems, nNodeI, nFaceI, nEleI] = extractMeshInfo(file_path);
FaceI = faces(:,2:end);
EleI = elems(:,2:end);

%% create stiffness matrices

%Young's modulus
EI = 1e7; %[Pa]

%poisson's ratio
nuI = 0.3;

%material constant % Isotropic 3D Hook's Law
c11 = EI * (1 - nuI)/(1 - 2 * nuI)/(1 + nuI);
c12 = EI * nuI /(1 - 2 * nuI)/(1 + nuI);
CI = [c11 c12 c12 0 0 0;
     c12 c11 c12 0 0 0;
     c12 c12 c11 0 0 0;
     0 0 0 (c11-c12)/2 0 0;
     0 0 0 0 (c11-c12)/2 0;
     0 0 0 0 0 (c11-c12)/2];

% core
KI = zeros(nNodeI * 3); % global stiffness matrix
VI = 0;
for i = 1 : nEleI
    
    x = NodeI(EleI(i, :), 1); % 4x1 vector containing all x values of an element. 
    y = NodeI(EleI(i, :), 2);
    z = NodeI(EleI(i, :), 3);
    
    x12 = x(1) - x(2); x21 = -x12;
    x23 = x(2) - x(3); x32 = -x23;
    x34 = x(3) - x(4); x43 = -x34;
    x13 = x(1) - x(3); x31 = -x13;
    x24 = x(2) - x(4); x42 = -x24;
    x14 = x(1) - x(4); x41 = -x14;
    
    y12 = y(1) - y(2); y21 = -y12;
    y23 = y(2) - y(3); y32 = -y23;
    y34 = y(3) - y(4); y43 = -y34;
    y13 = y(1) - y(3); y31 = -y13;
    y24 = y(2) - y(4); y42 = -y24;
    y14 = y(1) - y(4); y41 = -y14;
        
    z12 = z(1) - z(2); z21 = -z12;
    z23 = z(2) - z(3); z32 = -z23;
    z34 = z(3) - z(4); z43 = -z34;
    z13 = z(1) - z(3); z31 = -z13;
    z24 = z(2) - z(4); z42 = -z24;
    z14 = z(1) - z(4); z41 = -z14;

    a1 = y42*z32 - y32*z42;
    a2 = y31*z43 - y34*z13;
    a3 = y24*z14 - y14*z24;
    a4 = y13*z21 - y12*z31;

    b1 = x32*z42 - x42*z32;
    b2 = x43*z31 - x13*z34;
    b3 = x14*z24 - x24*z14;
    b4 = x21*z13 - x31*z12;

    c1 = x42*y32 - x32*y42;
    c2 = x31*y43 - x34*y13;
    c3 = x24*y14 - x14*y24;
    c4 = x13*y21 - x12*y31;      
    
    Ve = (1/6)*(x21*(y23*z34 - y34*z23) + x32*(y34*z12 - y12*z34) + x43*(y12*z23 - y23*z12));
    B = [a1,  0,  0, a2,  0,  0, a3,  0,  0, a4,  0,  0;
          0, b1,  0,  0, b2,  0,  0, b3,  0,  0, b4,  0;
          0,  0, c1,  0,  0, c2,  0,  0, c3,  0,  0, c4;
         b1, a1,  0, b2, a2,  0, b3, a3,  0, b4, a4,  0;
          0, c1, b1,  0, c2, b2,  0, c3, b3,  0, c4, b4;
         c1,  0, a1, c2,  0, a2, c3,  0, a3, c4,  0, a4;
         ]/6/Ve; % Strain matrix from linear displacement mode (constant strain tetrahedron). Not large deformation. 
    ke = Ve * B' * CI * B;
    
    idx = EleI(i, :);
    for j = 1:4
        for k = 1:4
            KI(3*idx(j)-2:3*idx(j), 3*idx(k)-2:3*idx(k)) = ...
            KI(3*idx(j)-2:3*idx(j), 3*idx(k)-2:3*idx(k)) + ke(3*j-2:3*j,3*k-2:3*k);
        end
    end
    VI = VI + Ve;
end

VI
clear x y z idx a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 B ...
    x12 x21 x23 x32 x34 x43 x13 x31 x24 x42 x14 x41...
    y12 y21 y23 y32 y34 y43 y13 y31 y24 y42 y14 y41...
    z12 z21 z23 z32 z34 z43 z13 z31 z24 z42 z14 z41

%% combine stiffness matrices

idxSurfI = unique(FaceI(:));
nSurfI = length(idxSurfI);
nInner = nNodeI - nSurfI;
KI = sparse(KI);

%% construct Laplacian matrices (outer surface only). 
[laplacianMatrixI, massMatrixI] = laplacianMatrix_cotan(NodeI, FaceI);

%%
clear i j k fileID ke m Ve tLine
save data.mat
