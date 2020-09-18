function [] = output2off(Node, Face, filename)
if ~isempty(filename)
    fileID = fopen(filename,'w');
    fprintf(fileID,'OFF\n');
    fprintf(fileID,'%d %d 0\n', size(Node,1), size(Face,1));
    
    for i = 1:size(Node,1)
        fprintf(fileID,'%12.8f %12.8f %12.8f\n', Node(i,1), Node(i,2), Node(i,3));
    end
    
    for i = 1:size(Face,1)
        fprintf(fileID,'%d %d %d %d\n', Face(i,1), Face(i,2), Face(i,3), Face(i,4));
    end
    fclose(fileID);
end
end

