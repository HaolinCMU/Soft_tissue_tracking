%% Extract the mesh information from mesh file. (for kidney model and aorta model. )
function [nodes, faces, elems, node_num, face_num, elem_num] = extractMeshInfo(file_path)
nodes = []; faces = []; elems = [];

f = fopen(file_path);

% Node info; [x, y, z]. 
node_num = str2num(fgetl(f));
for i = 1 : node_num
    coord_temp = str2num(fgetl(f));
    nodes = [nodes;coord_temp];
end

nodes = round(nodes, 6, 'significant');

% Element info; [region; ind1, ind2, ind3, ind4]. 
elem_num = str2num(fgetl(f));
for i = 1 : elem_num
    elem_temp = str2num(fgetl(f));
    elems = [elems;elem_temp];
end

% Face info; [region; ind1, ind2, ind3]. 
face_num = str2num(fgetl(f));
for i = 1 : face_num
    face_temp = str2num(fgetl(f));
    faces = [faces;face_temp];
end

fclose(f);

end

