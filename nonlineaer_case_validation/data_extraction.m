close all; clear all; clc

%% set parpool %%
% poolobj = parpool(2);
poolobj = parpool(4);


%% Extract stress & displacement information from multiple files %%
recordKeys_in = [11, 107, 8];
abqDir_in = "C:\Users\13426\Desktop\soft_tissue_tracking\code\ANN\nonlinear\inp_test";
sub_dir = fullfile(abqDir_in, '*.fil');
dat = dir(sub_dir);
file_path = abqDir_in;
file_path_stress = strcat(file_path, "\stress\");
file_path_coord = strcat(file_path, "\coor\");
% file_path_gaussian = strcat(file_path, "\gaussian\");

% for i = 1:length(dat)
i = 1;
parfor i = 1:length(dat)
    disp(strcat("Node file ", int2str(i), " - extracting started. "));
    disp(strcat("Node file ", int2str(i), " - extracting running... "));
    
    filFile_in = fullfile(dat(i).name);
    % export tensor from node file %
    Rec = fil2Str(strcat(abqDir_in, '\', filFile_in));
    out_stress = Rec11(Rec);
    out_coor = Rec107(Rec);
%     out_gaussian = Rec8(Rec);
    namePart = erase(dat(i).name, '.fil');
    
    n0 = 1000000;
    limit_1 = floor(length(out_stress)/n0);
    limit_2 = floor(length(out_coor)/n0);
%     limit_3 = floor(length(out_gaussian)/n0);
    
    k = 0;
    for j = 1:(limit_1+1)
        fileName_stress = strcat(file_path_stress, namePart, "_stress_", int2str(j), ".csv");
%         fileName_stress = strcat(file_path_stress, namePart, "_stress_", int2str(j), ".txt");
        if (j == limit_1+1)
            csvwrite(fileName_stress, out_stress(k*n0+1:length(out_stress),:));
%             save(fileName_stress, out_stress(k*n0+1:length(out_stress),:));
        else
            csvwrite(fileName_stress, out_stress(k*n0+1:(k+1)*n0,:));
%             save(fileName_stress, out_stress(k*n0+1:(k+1)*n0,:));
        end
        k = k + 1;
    end
    
    disp(strcat("Node file ", int2str(i), " - stress extracting completed. "));
    
%     k = 0;
%     for m = 1:(limit_3+1)
%         fileName_gaussian = strcat(file_path_gaussian, namePart, "_gaussian_", int2str(m), ".csv");
% %         fileName_gaussian = strcat(file_path_gaussian, namePart, "_gaussian_", int2str(m), ".txt");
%         if (m == limit_3+1)
%             csvwrite(fileName_gaussian, out_gaussian(k*n0+1:length(out_gaussian),:));
% %             save(fileName_gaussian, out_gaussian(k*n0+1:length(out_gaussian),:));
%         else
%             csvwrite(fileName_gaussian, out_gaussian(k*n0+1:(k+1)*n0,:));
% %             save(fileName_gaussian, out_gaussian(k*n0+1:(k+1)*n0,:));
%         end
%         k = k + 1;
%     end
%     
%     disp(strcat("Node file ", int2str(i), " - gaussian extracting completed. "));
    
    k = 0;
    for l = 1:(limit_2+1)
        fileName_coord = strcat(file_path_coord, namePart, "_coor_", int2str(l), ".csv");
%         fileName_coord = strcat(file_path_coord, namePart, "_coor_", int2str(l), ".txt");
        if (l == limit_2+1)
            csvwrite(fileName_coord, out_coor(k*n0+1:length(out_coor),:));
%             save(fileName_coord, out_coor(k*n0+1:length(out_coor),:));
        else
            csvwrite(fileName_coord, out_coor(k*n0+1:(k+1)*n0,:));
%             save(fileName_coord, out_coor(k*n0+1:(k+1)*n0,:));
        end
        k = k + 1;
    end
    
    disp(strcat("Node file ", int2str(i), " - coord extracting completed. "));
    disp(strcat("Node file ", int2str(i), " - all completed. "));
end

% close the parpool %
delete(poolobj);

disp("All files are completed. ");
