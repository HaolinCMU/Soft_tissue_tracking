function [laplacianMatrix,massMatrix] = laplacianMatrix_cotan(Node, Face)
disp('Start constructing cotangent laplacian matrix')
tic

nFace = size(Face, 1);
idxSurf = unique(Face(:));
nIdxSurf = length(idxSurf); 

%compute cotan weight for each vertex in a single triangle
cotanList = zeros(nFace, 3);
massMatrix = zeros(nIdxSurf, 1);
laplacianMatrix = zeros(nIdxSurf);

for i = 1 : nFace
    tri = Face(i, :);
    
    t1 = tri(1);
    t2 = tri(2);
    t3 = tri(3);
    v1 = Node(t1, :);
    v2 = Node(t2, :);
    v3 = Node(t3, :);
    
    area = 0.5 * norm(cross(v2 - v1, v3 - v1));
    a = norm(v3 - v2);
    b = norm(v1 - v3);
    c = norm(v2 - v1);
    
    cotanList(i, 1) = (b * b + c * c - a * a) / 4 / area;
    cotanList(i, 2) = (a * a + c * c - b * b) / 4 / area;
    cotanList(i, 3) = (b * b + a * a - c * c) / 4 / area;
    
    idxT = [find(idxSurf == t1);
               find(idxSurf == t2);
               find(idxSurf == t3)];
    
    massMatrix(idxT(1)) = massMatrix(idxT(1)) + area / 3;
    massMatrix(idxT(2)) = massMatrix(idxT(2)) + area / 3; % area/3 is an approximation? 
    massMatrix(idxT(3)) = massMatrix(idxT(3)) + area / 3; % for each node, add one third of the triangle's area. 
    
    cotan = cotanList(i,:);
    idx = [1,2,3,1,2];
    for j  = 1:3
        laplacianMatrix(idxT(idx(j)), idxT(idx(j))) = ...
            laplacianMatrix(idxT(idx(j)), idxT(idx(j))) - 0.5 * (cotan(idx(j+1)) + cotan(idx(j+2)));
        laplacianMatrix(idxT(idx(j)), idxT(idx(j+1))) = ...
            laplacianMatrix(idxT(idx(j)), idxT(idx(j+1))) + 0.5 * cotan(idx(j+2));
        laplacianMatrix(idxT(idx(j)), idxT(idx(j+2))) = ...
            laplacianMatrix(idxT(idx(j)), idxT(idx(j+2))) + 0.5 * cotan(idx(j+1)); 
    end    
end

% %expand laplacian matrix to 3 dimension
% temp = [repelem(laplacianMatrix,3,1),zeros(3*nIdxSurf, 2*nIdxSurf)];
% for i = 1:(3*nIdxSurf)
%     templine = zeros(1,3*nIdxSurf);
%     for j = 1:nIdxSurf
%         if temp(i,j) ~= 0
%             templine((j-1)*3+1) = temp(i,j);
%         end
%     end
%     temp(i,:) = templine;
% end
% for i = 1:nIdxSurf
%     for j = 1:3
%         temp(3*(i-1)+j,:) = [zeros(1,j-1),temp(3*(i-1)+j,1:3*nIdxSurf-j+1)];
%     end
% end
% 
% laplacianMatrix = temp;
% 
% %expand massMatrix to 3D diagonal matrix
% massMatrix = repelem(massMatrix,3,1);
massMatrix = diag(massMatrix);

disp('Finish constructing Laplacian')
toc
end


