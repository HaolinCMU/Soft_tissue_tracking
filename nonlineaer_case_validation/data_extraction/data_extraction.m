close all; clear all; clc

%% set parpool %%
poolobj = parpool(4);


%% Initialize paths and create folders %%
recordKeys_in = [11, 107, 8];
file_path = "C:\Users\13426\Desktop\soft_tissue_tracking\code\ANN\nonlinear\inp_files";
abqDir_in = strcat(file_path, "\fil");
sub_dir = fullfile(abqDir_in, '*.fil');
dat = dir(sub_dir);
file_path_stress = strcat(file_path, "\stress\");
file_path_coord = strcat(file_path, "\coor\");

extracted_folder_path = strcat(abqDir_in, "\extracted");
if ~isfolder(extracted_folder_path)
    mkdir(extracted_folder_path);
end


%% Extraction loop %%
i = 1;
parfor i = 1:length(dat)
    disp(strcat("Node file ", int2str(i), " - extracting started. "));
    disp(strcat("Node file ", int2str(i), " - extracting running... "));
    
    filFile_in = fullfile(dat(i).name);
    file_path_temp = strcat(abqDir_in, '\', filFile_in);
    Rec = fil2Str(file_path_temp);
    out_stress = Rec11(Rec);
    out_coor = Rec107(Rec);
    namePart = erase(dat(i).name, '.fil');
    
    n0 = 1000000;
    limit_1 = floor(length(out_stress)/n0);
    limit_2 = floor(length(out_coor)/n0);
    
    k = 0;
    for j = 1:(limit_1+1)
        fileName_stress = strcat(file_path_stress, namePart, "_stress_", int2str(j), ".csv");
        if (j == limit_1+1)
            csvwrite(fileName_stress, out_stress(k*n0+1:length(out_stress),:));
        else
            csvwrite(fileName_stress, out_stress(k*n0+1:(k+1)*n0,:));
        end
        k = k + 1;
    end
    
    disp(strcat("Node file ", int2str(i), " - stress extracting completed. "));

    k = 0;
    for l = 1:(limit_2+1)
        fileName_coord = strcat(file_path_coord, namePart, "_coor_", int2str(l), ".csv");
        if (l == limit_2+1)
            csvwrite(fileName_coord, out_coor(k*n0+1:length(out_coor),:));
        else
            csvwrite(fileName_coord, out_coor(k*n0+1:(k+1)*n0,:));
        end
        k = k + 1;
    end
    
    disp(strcat("Node file ", int2str(i), " - coord extracting completed. "));
    disp(strcat("Node file ", int2str(i), " - all completed. "));
    
    % Move extracted file to the new-created folder. 
    movefile(file_path_temp, extracted_folder_path);
end


%% Close the parpool %%
delete(poolobj);
disp("All files are completed. ");
