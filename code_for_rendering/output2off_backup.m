function  [V,Tri,fcolors] = output2off( v, ele, idxSurf, filename,color,force)

%compute surface triangle list
triList = [];
for i = 1:size(ele,1)
    Lia = ismember(ele(i,:),idxSurf);
    if sum(Lia) == 3
        currentTri = ele(i,Lia);
        triList = [triList;currentTri]; 
    end
end

idxV = unique(triList);
V = v(idxV,:);
Tri = triList;%usage
fcolors = force;
for i = 1:size(Tri,1)
    v1 = V(Tri(i,1),:);
    v2 = V(Tri(i,2),:);
    v3 = V(Tri(i,3),:);
    cro = cross(v2-v1,v3-v1);
    if dot(v1,cro) < 0
       temp = Tri(i,1);
       Tri(i,1) = Tri(i,2);
       Tri(i,2) = temp;
    end
end

if ~isempty(filename)
    
    fileID = fopen(filename,'w');
    fprintf(fileID,'COFF\n');
    fprintf(fileID,'%d %d 0\n', size(V,1), size(Tri,1));
    if isempty(force)
        for i = 1:size(V,1)
            fprintf(fileID,'%12.8f %12.8f %12.8f %s\n', V(i,1),V(i,2),V(i,3), color);
        end
    else
        fmin = min(force);
        fran = range(force);
        for i = 1:size(V,1)
            fcolor = (force(i) - fmin) / fran * 5.0;
            if fcolor < 0.5
                fcolor = fcolor + 0.5;
                fcolor = [0.0 0.0 fcolor];
            elseif fcolor < 1.5
                fcolor = fcolor - 0.5;
                fcolor = [0.0 fcolor 1.0];
            elseif fcolor < 2.5 
                fcolor = 2.5 - fcolor;
                fcolor = [0.0 1.0 fcolor];
            elseif fcolor < 3.5
                fcolor = fcolor - 2.5;
                fcolor = [fcolor 1.0 0.0];
            elseif fcolor < 4.5
                fcolor = 4.5 - fcolor;
                fcolor = [1.0 fcolor 0.0];
            else
                fcolor = 5.5 - fcolor;
                fcolor = [fcolor 0.0 0.0];
            end
            fprintf(fileID,'%12.8f %12.8f %12.8f %s\n', V(i,1),V(i,2),V(i,3),num2str(fcolor,'%0.6f '));
        end
    end


    for i = 1:size(Tri,1)
        fprintf(fileID,'3 %d %d %d\n', Tri(i,1)-1, Tri(i,2)-1, Tri(i,3)-1);
    end
    fclose(fileID);
end
end

