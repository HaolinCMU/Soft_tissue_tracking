close all; clear all; clc

%% Read geometric model files (same directory). 
geometry_folder = 'geometry_files';
node_file_path = strcat(geometry_folder, '\core.node');
elem_file_path = strcat(geometry_folder, '\core.ele');
face_file_path = strcat(geometry_folder, '\core.face');
edge_file_path = strcat(geometry_folder, '\core.edge');
deformed_config_file = "ANN_benchmark_results.mat"; % Containing deformed configurations of the geometry. 

isDeformedOn = 1; % 0: Not deformed geometry; 1: deformed geometry.

isReconstructOn = 0; % 0: ANN-reconstructed configuration; 1: FEM-generated configuration. 
sample_NO = 80; % NO. of the sample. [1,100]. 


%% Extract node information. 
f = fopen(node_file_path);
node_info = str2num(fgetl(f));
node_num = node_info(1);

node_list = [];
for i = 1:node_num
    num_list_temp = str2num(fgetl(f));
    coord_temp = num_list_temp(2:4);
    node_list = [node_list;coord_temp];
end
fclose(f);

if isDeformedOn
    load(deformed_config_file);
    
    label_configs = [];
    reconstruct_configs = [];
    
    for i = 1:size(test_deformation_label, 2)
        deform_label_temp = reshape(test_deformation_label(:, i), 3, [])'; % nNode * 3 matrix. Size 2: xyzxyz...
        label_configs(:,:,i) = node_list + deform_label_temp;
    end
    
    for i = 1:size(test_deformation_reconstruct, 2)
        deform_reconstruct_temp = reshape(test_deformation_reconstruct(:, i), 3, [])'; % nNode * 3 matrix. Size 2: xyzxyz...
        reconstruct_configs(:,:,i) = node_list + deform_reconstruct_temp;
    end
    
    if isReconstructOn
        node_list = reconstruct_configs(:,:,sample_NO);
    else
        node_list = label_configs(:,:,sample_NO);
    end   
end


%% Extract face information
f = fopen(face_file_path);
face_info = str2num(fgetl(f));
face_num = face_info(1);

face_list = [];
for i = 1:face_num
    face_list_temp = str2num(fgetl(f));
    length_temp = length(face_list_temp(2:4));
    face_list_temp = [length_temp, face_list_temp(2:4)];
    face_list = [face_list;face_list_temp];
end
fclose(f);


%% Write .off file. 
write_path = 'model_' + string(sample_NO) + '.off';
output2off(node_list, face_list, write_path);


%% Visualize .off file
figure
trisurf(face_list(:,2:end)+1, node_list(:, 1), node_list(:, 2), node_list(:, 3), 'FaceColor', 'none', 'EdgeColor', [0.25 0.25 1.0]); 









