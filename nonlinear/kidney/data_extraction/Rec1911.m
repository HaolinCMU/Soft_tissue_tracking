%
% ABAQUS output request definition output to MATLAB
% 
% Syntax
%     [#OutFlag#,#SetName#,#EleType#]=Rec1911(#Rec#,zeros(1,0),zeros(1,0))
%
% Description
%     Read output request definition from the results file generated from
%     the ABAQUS finite element software. The record key for output request
%     definition is 1911. See section ''Results file output format'' in
%     ABAQUS Analysis User's manual for more details.
%     The following option with parameter has to be specified in the ABAQUS
%     input file for the results file to be created:
%         *FILE FORMAT, ASCII
%     Dummy input arguments are required for execution of this function, as
%     illustrated in the 'Input parameters' section below. These are used
%     for memory allocation issues and proper initialization of internal
%     variables.
%     
% Input parameters
%     REQUIRED:
%     #Rec# (string) is a one-row string containing all the data inside the
%         Abaqus results file. The results file is generated by Abaqus
%         after the analysis has been completed.
%     DUMMY (in the following order, after REQUIRED):
%     zeros(1,0)
%     zeros(1,0)
% 
% Output parameters
%     #OutFlag# ([#n# x 1]) contains a flag for element-based output (0),
%         nodal output (1), modal output (2), or element set energy output
%         (3).
%     #SetName# ([#n# x 8]) is a string array having 8 characters per row.
%         Each row contains the set name (node or element set) used in the
%         request (A8 format). This attribute is blank if no set was
%         specified.
%     #EleType# ([#n# x 8]) is a string array having 8 characters per row.
%         Each row contains the element type (only for element output, A8
%         format).
%     #n# is the number of output request definitions.
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
