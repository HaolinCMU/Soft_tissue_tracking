%
% Find the category of a record key
%
% Syntax
%     #cat#=catRecKey(#recordKey#)
%
% Description
%     This function returns a string showing the category to which a record
%     key belongs. The category is defined based on the type of attributes
%     that a record contains (e.g. integer, string, floating point number,
%     etc) as well as their order, which are associated with the record key
%     provided by the user. The categories of the record keys are generally
%     the following:
%         1) Category 1: the record contains only floating point numbers as
%            attributes
%         2) Category 2: the record contains an integer at location 3, and
%            floating point numbers at all the rest locations.
%         3) Category 3: the record contains floating point numbers and
%            strings as attributes.
%     Based on the category of the record key, the suitable post-processing
%     function is used for the extraction of the results. For category 1,
%     the function readFilCat1.m is used, for category 2 the function
%     readFilCat2.m is used, for category 3 the function readFilCat3.m is
%     used, etc.
%
% Input parameters
%     #recordKey# (double) is the key of the record the category of which
%         is requested. It must be integer, otherwise an error is issued.
%
% Output parameters
%     #cat# (string) is the category of the record key. If the input field
%         output identifier does not belong to any of the three categories
%         or is invalid, an empty string is returned.
%
% Example
%     recordKey=45;
%     cat=catRecKey(recordKey)
%
% _________________________________________________________________________
% Abaqus2Matlab - www.abaqus2matlab.com
% Copyright (c) 2017 by George Papazafeiropoulos
%
% If using this toolbox for research or industrial purposes, please cite:
% G. Papazafeiropoulos, M. Muniz-Calvente, E. Martinez-Paneda.
% Abaqus2Matlab: a suitable tool for finite element post-processing.
% Advances in Engineering Software. Vol 105. March 2017. Pages 9-16. (2017) 
% DOI:10.1016/j.advengsoft.2017.01.006

% Built-in function (Matlab R2012b)
