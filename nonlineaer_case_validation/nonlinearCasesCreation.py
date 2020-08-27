# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 13:08:16 2020

@author: haolinl
"""

import copy
import os

import numpy as np
import scipy.io # For extracting data from .mat file


class inputFileGenerator(object):
    """
    Generate input file for Abaqus. 

    Unit: 
        Length: m
        Force: N
        Pressure: Pa
    """

    def __init__(self, data_file_name, write_path, fix_indices_list):
        """
        Initialize parameters. 

        Parameters: 
        ----------
        
        """
        
        # Data & Variables. 
        self.data_file_name = data_file_name
        self.data_mat = scipy.io.loadmat(self.data_file_name)
        self.surface_nodes_num = self.data_mat["nSurfI"][0,0]
        self._inputFile_lines_total = []
        self.writePath = write_path

        # Header. 
        self._header = ["*Heading"]

        # Part definition. 
        self._part_name = "part-1"
        self._material_name = "tissue"
        self._part_initial = ["*Part, name={}".format(self._part_name)] # Total list of Part definition. 
        self._node = ["*Node"]
        self._elem = ["*Element, type=C3D4"]
        self._nset_all = []
        self._elset_all = []
        self._section = ["*Solid Section, elset=allElems, material={}".format(self._material_name),
                         ","]
        self._part_end = ["*End Part"]
        self._part = self.generatePart()

        # Assembly definition. 
        self._assembly_name = "assembly-1"
        self._instance_name = "instance-1"
        self._assembly_initial = ["*Assembly, name={}".format(self._assembly_name)] # Total list of Assembly definition. 
        self._instance = ["*Instance, name={}, part={}".format(self._instance_name, self._part_name),
                          "*End Instance"]

        self._fix_nset_name = "fix"
        self._fix_indices_list = fix_indices_list
        self._fix_nset = self.generateNset(self._fix_indices_list, self._fix_nset_name, self._instance_name) # Nset definition of fix BC. 
        self.loads_num = 3 # For initial testing.
        self._load_name_list = []
        self._load_nsets = [] # Nset definition of loads. 

        self._nset_boundary = [] # All nsets definitions in assembly. Boundary conditions
        self._assembly_end = ["*End Assembly"]
        self._assembly = self.generateAssembly()

        # Material. 
        self._modulus = 10000000 # Young's modulus. Unit: Pa. 
        self._poisson_ratio = 0.3 # Poisson's ratio. 
        self._material = ["*Material, name={}".format(self._material_name),
                          "*Elastic",
                          "{}, {}".format(self._modulus, self._poisson_ratio)]
        
        # Boundary condition. 
        self._boundary_initial = ["*Boundary"]
        self._boundary = self.generateBoundaryCondition_fixAll()

        # Step settings. 
        self.autoIncrementNum = 5000 # Int. The maximum increment number of the AutoSolver. 
        self.initIncrem = 0.001 # FLoat. The initial length of the increment (for fixed-step, this is also the length per increm). 
        self.minIncrem = 1e-20 # Float. The minimum increment length for the AutoSolver (ueless for the StaticSolver). 
        self.maxIncrem = 1.0 # Float. The maximum increment length for the AutoSolver (useless for the StaticSovler). 
        self.totalTime = 1.0 # Float. The total time for one simulation step. 
        self.frameNum = 1 # Int. The number of frames intending to extract from the nodal file. 
        self.freq = int(self.autoIncrementNum / self.frameNum) # Int. The data frame extraction frequency (also refers to the number of increments. Extract one frame per "self.freq" increments). Especially for StaticSolver case.  
        
        self._step = ["*Step, name=step-1, nlgeom=YES, inc={}".format(self.autoIncrementNum),
                      "*Static",
                      "{}, {}, {}, {}".format(self.initIncrem, self.totalTime, 
                                              self.minIncrem, self.maxIncrem)] # Auto solver. 
        self._step_end = ["*End Step"]

        # Load settings.
        self._load_scale = 50 # Case and BC specific. Unit: N. 
        self._load = self.generateLoadSetting()


        # Rest settings. 
        self._restart = ["*Restart, write, frequency=0"]
        self._output = ["*Output, field, variable=PRESELECT",
                        "*Output, history, variable=PRESELECT"]
        self._fil = ["*FILE FORMAT, ASCII",
                     "*node file, frequency={}".format(self.freq),
                     "U, COORD",
                     "*El file, frequency={}".format(self.freq),
                     "S, COORD"]
        self._resSettings = self._restart + self._output + self._fil
                
    
    def readFile(self, read_path):
        """
        Read files from specific path. 

        Parameters:
        ----------
            read_path: String. 
                Path of the original inp file.

        Return:
        ----------
            lines: List of strings. 
                The list of lines from the file. 
        """
        
        with open(read_path, "rt") as f: lines = f.read().splitlines()
        
        return lines
    

    def writeFile(self):
        """
        Write 'self.write_lines' into new inp file. 

        """
        
        self._inputFile_lines_total = (self._header + self._part + self._assembly + 
                                       self._material + self._boundary + self._step + 
                                       self._load + self._resSettings + self._step_end)

        content = '\n'.join(self._inputFile_lines_total)
        
        with open(self.writePath, 'w') as f: f.write(content)
    

    def generatePart(self):
        """

        """

        self.generateNodes()
        self.generateElements()

        # Generate all element elset. 
        allElem_list, allElem_list_name = [], "allElems"
        for i in range(len(self._elem[1:])): allElem_list.append(str(i+1))
        self._elset_all = self.generateElset(allElem_list, allElem_list_name)

        # Generate Section. 
        self._section = self.generateSection(allElem_list_name, self._material_name)

        # Collection. 
        return (self._part_initial + self._node + self._elem + self._elset_all + 
                      self._section + self._part_end)


    def generateNodes(self):
        """

        """

        node_mat = self.data_mat["NodeI"]

        for i in range(node_mat.shape[0]):
            node_list_temp = ["{}".format(i+1)]
            node_list_temp += [str(coord) for coord in list(node_mat[i,:])]
            self._node.append(', '.join(node_list_temp))
    
    
    def generateElements(self):
        """

        """

        elem_mat = self.data_mat["EleI"]

        for i in range(elem_mat.shape[0]):
            elem_list_temp = ["{}".format(i+1)]
            elem_list_temp += [str(ind) for ind in list(elem_mat[i,:])]
            self._elem.append(', '.join(elem_list_temp))
    

    def generateNset(self, node_list, nset_name, instance_name=None):
        """
        """

        if instance_name == None: nset = ["*Nset, nset={}".format(nset_name)]
        else: nset = ["*Nset, nset={}, instance={}".format(nset_name, instance_name)]

        nset_line_temp, nset_string_temp = [], None

        for i, ind in enumerate(node_list):
            nset_line_temp.append(str(ind))

            if (i+1) % 10 == 0:
                nset_string_temp = ', '.join(nset_line_temp)
                nset.append(copy.deepcopy(nset_string_temp))
                nset_line_temp, nset_string_temp = [], None
        
        nset_string_temp = ', '.join(nset_line_temp)
        nset.append(copy.deepcopy(nset_string_temp))

        return nset
    
    
    def generateElset(self, elem_list, elset_name, instance_name=None):
        """
        """

        if instance_name == None: elset = ["*Elset, elset={}".format(elset_name)]
        else: elset = ["*Elset, elset={}, instance={}".format(elset_name, instance_name)]

        elset_line_temp, elset_string_temp = [], None

        for i, ind in enumerate(elem_list):
            elset_line_temp.append(str(ind))

            if (i+1) % 10 == 0:
                elset_string_temp = ', '.join(elset_line_temp)
                elset.append(copy.deepcopy(elset_string_temp))
                elset_line_temp, elset_string_temp = [], None

        elset_string_temp = ', '.join(elset_line_temp)
        elset.append(copy.deepcopy(elset_string_temp))

        return elset
    
    
    def generateSection(self, elset_name, material_name):
        """
        """

        section = ["*Solid Section, elset={}, material={}".format(elset_name, material_name),
                   ","]

        return section
    
    
    def generateAssembly(self):
        """

        """

        # Generate "self.loads_num" nsets, each of which has 1 node. 
        for i in range(self.loads_num): 
            load_name_temp = "Load-{}".format(i+1)
            self._load_name_list.append(load_name_temp)
            load_posi_index_temp = np.random.randint(1, self.surface_nodes_num+1) # Rnadomly chosen a surface node to apply load F(x, y, z). Indexed from 1. 
            self._load_nsets += self.generateNset([load_posi_index_temp], load_name_temp, 
                                                  self._instance_name)
        
        self._nset_boundary = self._nset_boundary + self._load_nsets + self._fix_nset

        return (self._assembly_initial + self._instance + self._nset_boundary + self._assembly_end)


    def generateBoundaryCondition_fixAll(self):
        """

        """

        BC_list_temp = []
        for i in range(6): # 6: 6 DOFs. 
            BC_list_temp.append("{}, {}, {}".format(self._fix_nset_name, i+1, i+1))
        
        return (self._boundary_initial + BC_list_temp)
    

    def generateLoadSetting(self):
        """

        """

        load_list = []

        for load_name in self._load_name_list: # Length: self._loads_num
            load_temp = ["*Cload"]

            for i in range(3): # 3: Three directions. 
                load_value_temp = (np.random.rand() * 2 - 1) * self._load_scale
                load_temp.append("{}, {}, {}".format(load_name, i+1, load_value_temp))
            
            load_list += copy.deepcopy(load_temp)
        
        return load_list


def main():
    """
    """

    inp_folder = "inp_files"
    sample_nums = 1000
    data_file_path = "data_head_and_neck.mat"
    fix_indices_list = [805, 815, 839] # Specify the node to fix. At least 3. Indexed from 1. 

    # Generate input file for Abaqus. 
    for i in range(sample_nums):
        if not os.path.isdir(inp_folder): os.mkdir(inp_folder)
        
        write_path = os.path.join(inp_folder, "{}.inp".format(str(i+1)))
        inputFile_temp = inputFileGenerator(data_file_path, write_path, fix_indices_list)
        inputFile_temp.writeFile()


if __name__ == "__main__":
    main()


