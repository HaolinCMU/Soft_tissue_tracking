# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 13:08:16 2020

@author: haolinl
"""

import copy
import os
import time

import numpy as np
import scipy.io # For extracting data from .mat file


class inputFileGenerator(object):
    """
    Generate input file for Abaqus. 

    Unit system: 
        Length: m
        Force: N
        Pressure: Pa
    """

    def __init__(self, data_file_name, write_path, fix_indices_list, node_variable_name, elem_variable_name):
        """
        Initialize parameters. 

        Parameters: 
        ----------
        data_file_name: String. 
            The file path of information of node, element, etc. 
        write_path: String. 
            The path to write the inp file. 
        fix_indices_list: List of ints. 
            The node indices to be fixed. 
        node_variable_name: String. 
            The variable name of the nodes matrix in the data file. 
        elem_variable_name: String. 
            The variable name of the elements matrix in the data file. 
        """
        
        # Data & Variables. 
        self.data_file_name = data_file_name
        self.data_mat = scipy.io.loadmat(self.data_file_name)
        self._surface_mat = self.data_mat["FaceI"]
        self._surface_nodes = self.data_mat["idxSurfI"]
        self._surface_nodes_num = self.data_mat["nSurfI"][0,0]
        self._triangle_nodes_list = []
        self._coupled_list = []
        self._node_variable_name = node_variable_name
        self._elem_variable_name = elem_variable_name
        self._inputFile_lines_total = []
        self.writePath = write_path

        # Header. 
        self._header = ["*Heading"]

        # Part definition. 
        self._part_name = "part-1"
        self._material_name = "tissue"
        self._part_initial = ["*Part, name={}".format(self._part_name)] # Total list of Part definition. 
        self._node = ["*Node"]
        self._elem = ["*Element, type=C3D10"] # Nonlinear tetrahedron. http://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node33.html#tennode. 
        self._nset_all = []
        self._elset_all = []
        self._section = ["*Solid Section, elset=allElems, material={}".format(self._material_name),
                         ","]
        self._part_end = ["*End Part"]
        self._new_node_list = []
        self._new_node_dict = {}
        self._node_num = None
        self._orig_node_num = None
        self._elem_num = None
        self._part = self.generatePart()

        # Assembly definition. 
        self._assembly_name = "assembly-1"
        self._instance_name = "instance-1"
        self._assembly_initial = ["*Assembly, name={}".format(self._assembly_name)] # Total list of Assembly definition. 
        self._instance = ["*Instance, name={}, part={}".format(self._instance_name, self._part_name),
                          "*End Instance"]

        self._ref_nodes_list = []
        
        self._fix_nset_name = "fix"
        self._fix_indices_list = fix_indices_list
        self._fix_nset = self.generateNset(self._fix_indices_list, self._fix_nset_name, self._instance_name) # Nset definition of fix BC. 
        self.loads_num = 3 # For initial testing.
        self._coupling_neighbor_layers = 1 # How deep does the neighborhood searching go. Default: 1. 
        self._rf_name_list = []
        self._rf_nset_name_list = []
        self._rf_nsets = []
        self._load_nsets = [] # Nset definition of loads. 

        self._surface_list = []
        self._coupling_list = []

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
        self.initIncrem = 0.001 # Float. The initial length of the increment (for fixed-step, this is also the length per increm). 
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
        self._load_scale = (0.0, 10.0) # Absolute range of the force. Case and BC specific. (min, max). Unit: N. 
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
        Write 'self.write_lines' into a new inp file. 

        """
        
        self._inputFile_lines_total = (self._header + self._part + self._assembly + 
                                       self._material + self._boundary + self._step + 
                                       self._load + self._resSettings + self._step_end)

        content = '\n'.join(self._inputFile_lines_total)
        
        with open(self.writePath, 'w') as f: f.write(content)
    

    def generatePart(self):
        """
        Generate part definition.  

        Returns:
        ----------
        The list collection of all sub-definition lists, including:
            part_initial: header part of "Part definition". 
            node: Node definition.
            elem: Element definition.
            elset_all: The elset containing all elements. For material definition specifically. 
            section: Section definition. 
            part_end: The endline of "Part definition". 
        """

        self.generateNodes(self.data_mat[self._node_variable_name], self._node)
        self.generateElements(self.data_mat[self._elem_variable_name], self._elem)

        self.nonlinearization()

        # Generate all element elset. 
        allElem_list, allElem_list_name = [], "allElems"
        for i in range(len(self._elem[1:])): allElem_list.append(str(i+1))
        self._elset_all = self.generateElset(allElem_list, allElem_list_name)

        # Generate Section. 
        self._section = self.generateSection(allElem_list_name, self._material_name)

        # Collection. 
        return (self._part_initial + self._node + self._elem + self._elset_all + 
                      self._section + self._part_end)


    def generateNodes(self, node_mat, target_node_list, specified_indices_list=[]):
        """
        Generate nodes information. 

        Parameters:
        ----------
            node_mat: 2D Array of ints. 
                The matrix containing the coordinates of the nodes to-be-defined under "*Node". 
            targer_node_list: List of strings. 
                The definition of node list. 
            specified_indices_list (optional): List of ints.
                List the indices of the input node list, following the exact order of the node_mat.  
                Default: []. 
        """

        for i in range(node_mat.shape[0]):
            if specified_indices_list == []: node_list_temp = ["{}".format(i+1)]
            else: node_list_temp = ["{}".format(specified_indices_list[i])]
            node_list_temp += [str(coord) for coord in list(node_mat[i,:])]
            target_node_list.append(', '.join(node_list_temp))
    
    
    def generateElements(self, elem_mat, target_elem_list, specified_indices_list=[]):
        """
        Generate elements information. 

        Parameters:
        ----------
            elem_mat: 2D Array of ints. 
                The matrix containing the indices of each element to-be-defined under "*Element". 
            targer_elem_list: List of strings. 
                The definition of element list. 
            specified_indices_list (optional): List of ints.
                List the indices of the input element list, following the exact order of the elem_mat.  
                Default: []. 
        """

        for i in range(elem_mat.shape[0]):
            if specified_indices_list == []: elem_list_temp = ["{}".format(i+1)]
            else: elem_list_temp = ["{}".format(specified_indices_list[i])]
            elem_list_temp += [str(ind) for ind in list(elem_mat[i,:])]
            target_elem_list.append(', '.join(elem_list_temp))
    

    def generateNset(self, node_list, nset_name, instance_name=None):
        """
        Generate node set information. 

        Parameters:
        ----------
        node_list: List of ints. 
            The list of nodes to be contained in the node list. 
        nset_name: String. 
            The name of the to-be-defined node list. 
        instance_name (optional): String. 
            The name of specified instance. 
            Only use in assembly definition. 
            Default: None. (Part cases)

        Returns:
        ----------
        nset: List of strings. 
            The definition of a specific nset. 

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
        Generate element set information.

        Parameters:
        ----------
        elem_list: List of ints. 
            The list of elements to be contained in the element list. 
        elset_name: String. 
            The name of the to-be-defined element list. 
        instance_name (optional): String. 
            The name of specified instance. 
            Only use in assembly definition. 
            Default: None. (Part cases)

        Returns:
        ----------
        elset: List of strings. 
            The definition of a specific elset.         
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
        Generate section information. 

        Parameters:
        ----------
        elset_name: String. 
            The name of the elset to be assigned a section. 
        material_name: String. 
            The name of defined material. 

        Returns:
        ----------
        section: List of strings. 
            The definition of section. 
        """

        section = ["*Solid Section, elset={}, material={}".format(elset_name, material_name),
                   ","]

        return section
    
    
    def generateAssembly(self):
        """
        Generate assembly definition. 

        Returns:
        ----------
        The list collection of all sub-definition lists, including:
            assenbly_initial: Header of the assembly definition. 
            instance: The instance definition. 
            nset_boundary: The definition of BC related node set. 
            asssenbly_end: The endline of assembly definition. 

        """

        # Generate "self.loads_num" nsets, each of which has 1 node. 
        for i in range(self.loads_num): 
            ref_name_temp = "rf-{}".format(i+1)
            ref_nset_name_temp = "rf-{}-nset".format(i+1)
            self._rf_name_list.append(ref_name_temp)
            self._rf_nset_name_list.append(ref_nset_name_temp)

            while(True):
                load_posi_index_temp = np.random.randint(1, self._surface_nodes_num+1) # Randomly chosen a surface node to apply load F(x, y, z). Indexed from 1. 
                if load_posi_index_temp not in self._fix_indices_list: break # The randomly generated index cannot be one of the fixed nodes. 
            
            # Generate assembly node definitions for reference points. 
            ref_node_list_temp = ["*Node"]
            ref_pt_coord_list_temp = [float(item) for item in self._node[load_posi_index_temp].split(',')[1:]]
            self.generateNodes(np.array(ref_pt_coord_list_temp).astype(float).reshape(1,-1), ref_node_list_temp, 
                               specified_indices_list=[i+1])
            self._ref_nodes_list += copy.deepcopy(ref_node_list_temp)
            rf_nset_list_temp = self._findCouplingNodes(load_posi_index_temp, self._coupling_neighbor_layers)

            # Generate reference point node sets. 
            self._load_nsets += self.generateNset([i+1], ref_name_temp)
            
            # Generate coupling constraint node sets. 
            self._rf_nsets += self.generateNset(rf_nset_list_temp, ref_nset_name_temp, 
                                                  self._instance_name)
        
        self.generateCoupling()
        self._nset_boundary = self._nset_boundary + self._load_nsets + self._rf_nsets + self._fix_nset + self._surface_list + self._coupling_list

        return (self._assembly_initial + self._instance + self._ref_nodes_list + self._nset_boundary + self._assembly_end)


    def generateCoupling(self):
        """
        Generate coupling constriants for concentrated forces application. 
        """

        for index, rf_name in enumerate(self._rf_nset_name_list):
            self._surface_list += ["*Surface, type=NODE, name={}_CNS_, internal".format(rf_name),
                                   "{}, 1.".format(rf_name)]
            self._coupling_list += ["*Coupling, constraint name={}, ref node={}, surface={}_CNS_".format(self._rf_name_list[index], 
                                                                                                         self._rf_name_list[index],
                                                                                                         rf_name),
                                    "*Kinematic"]
    

    def _findCouplingNodes(self, rf_node_ind, neighbor_layers):
        """
        Find the immediate neighbors of each specified node index. 

        Parameters:
        ----------
            rf_node_ind: Int. 
                The index of target node. 

        Returns:
        ----------
            rf_nset_list: List of ints (duplicated items removed). 
                "rf_node_ind"'s corresponding immediate neighbor nodes set. 
        """

        rf_nset_list, new_nodes_list, searched_nodes_list = [rf_node_ind], [rf_node_ind], []

        for j in range(neighbor_layers):
            for ind_temp in new_nodes_list:
                for i in range(len(self._triangle_nodes_list)):
                    if ind_temp in self._triangle_nodes_list[i]: 
                        rf_nset_list += copy.deepcopy(self._triangle_nodes_list[i])
                    else: continue
            
            searched_nodes_list += copy.deepcopy(new_nodes_list)
            rf_nset_list = list(set(copy.deepcopy(rf_nset_list)))
            new_nodes_list = [ind for ind in rf_nset_list if ind not in searched_nodes_list]

        # Avoid assigning same nodes to different coupled node sets. 
        for ind in rf_nset_list:
            if ind in self._coupled_list: rf_nset_list.remove(ind)
            else: self._coupled_list.append(ind)

        return rf_nset_list
    

    def generateBoundaryCondition_fixAll(self):
        """
        Generate fix boundary condition. 

        Returns:
        ----------
        The list collection of all sub-definition lists, including:
            boundary_initial: Header of boundary condition definition. 
            BC_list_temp: The detailed BC definition of boundary conditions. 
        """

        BC_list_temp = []
        for i in range(6): # 6: 6 DOFs (disp. + rot.); 3: 3 DOFs (disp.). 
            BC_list_temp.append("{}, {}, {}".format(self._fix_nset_name, i+1, i+1))
        
        return (self._boundary_initial + BC_list_temp)
    

    def generateLoadSetting(self):
        """
        Generate load information. 

        Returns:
        ----------
        load_list: List of strings. 
            Definition of concentrated forces. 

        """

        load_list = []

        for rf_name in self._rf_name_list: # Length: self._loads_num
            load_temp = ["*Cload, op=NEW"]

            for i in range(3): # 3: Three directions. 
                load_value_temp = (np.random.rand() * 2 - 1) * abs(self._load_scale[1] - self._load_scale[0])

                if load_value_temp < 0: load_value_temp -= self._load_scale[0]
                else: load_value_temp += self._load_scale[0]

                load_temp.append("{}, {}, {}".format(rf_name, i+1, load_value_temp))
            
            load_list += copy.deepcopy(load_temp)
        
        return load_list
    
    
    def _computeMidPoint(self, ind_1, ind_2):
        """
        Compute the mid-point of the edge. 

        Parameters:
        ----------
            ind_1: Int. 
                The first index of the node pair. Indexed from 1. 
            ind_2: Int. 
                The second index of the node pair. Indexed from 1.
        
        Returns:
        ----------
            ind_mid: Int. 
                The index of the self._node. Index from 1. 
        """

        key_string_temp_1, key_string_temp_2 = "{}_{}".format(ind_1, ind_2), "{}_{}".format(ind_2, ind_1)

        if key_string_temp_1 in self._new_node_dict.keys(): return self._new_node_dict[key_string_temp_1]
        elif key_string_temp_2 in self._new_node_dict.keys(): return self._new_node_dict[key_string_temp_2]
        
        else:
            coord_temp_1 = np.array(self._node[ind_1].split(',')[1:]).astype(float).reshape(1,-1)
            coord_temp_2 = np.array(self._node[ind_2].split(',')[1:]).astype(float).reshape(1,-1)

            coord_temp_mid = (coord_temp_1 + coord_temp_2) / 2.0
            coord_mid_list = [str(item) for item in list(coord_temp_mid[0])]

            self._node_num = len(self._node)
            new_node_def_list_temp = copy.deepcopy([str(self._node_num)])
            new_node_def_list_temp += copy.deepcopy(coord_mid_list)
            self._node.append(', '.join(new_node_def_list_temp))
            self._new_node_list.append(', '.join(new_node_def_list_temp))

            self._new_node_dict[key_string_temp_1] = self._node_num
            self._new_node_dict[key_string_temp_2] = self._node_num

            return self._node_num
    

    def insertNode(self):
        """
        Insert one node (at the mid-point) of each edge. 
        Create C3D10 element structure. 
        """

        for index, elem_def_string in enumerate(self._elem[1:]):
            elem_node_list_temp = [int(ind) for ind in elem_def_string.split(',')[1:]]
            
            # Obtain the mid-point index in order. Assume tetrahedral element (C3D4). 
            mid_pt_ind_5 = self._computeMidPoint(elem_node_list_temp[0], elem_node_list_temp[1])
            mid_pt_ind_6 = self._computeMidPoint(elem_node_list_temp[1], elem_node_list_temp[2])
            mid_pt_ind_7 = self._computeMidPoint(elem_node_list_temp[0], elem_node_list_temp[2])
            mid_pt_ind_8 = self._computeMidPoint(elem_node_list_temp[0], elem_node_list_temp[3])
            mid_pt_ind_9 = self._computeMidPoint(elem_node_list_temp[1], elem_node_list_temp[3])
            mid_pt_ind_10 = self._computeMidPoint(elem_node_list_temp[2], elem_node_list_temp[3])

            elem_new_def_list_temp = [str(mid_pt_ind_5), 
                                      str(mid_pt_ind_6),
                                      str(mid_pt_ind_7),
                                      str(mid_pt_ind_8),
                                      str(mid_pt_ind_9),
                                      str(mid_pt_ind_10)]
            
            # Redefine the new C3D10 element in order. 
            elem_def_list_temp = copy.deepcopy(elem_def_string.split(',')) + copy.deepcopy(elem_new_def_list_temp)
            elem_def_string_temp = ', '.join(elem_def_list_temp)

            self._elem[index+1] = copy.deepcopy(elem_def_string_temp)

    
    def _triangleNodesCollection(self):
        """
        Collect all the nodes on each triangle (surface). 
        Need to be implemented after "self.insertNode()". 
        """

        for i in range(self._surface_mat.shape[0]):
            tri_temp = self._surface_mat[i,:]

            # Assuming all triangles on the surface of geometry. 
            middle_pts_list_temp = [self._computeMidPoint(tri_temp[0], tri_temp[1]),
                                    self._computeMidPoint(tri_temp[0], tri_temp[2]),
                                    self._computeMidPoint(tri_temp[1], tri_temp[2])]

            triangle_nodes_list_temp = list(copy.deepcopy(tri_temp)) + copy.deepcopy(middle_pts_list_temp)
            self._triangle_nodes_list.append(copy.deepcopy(triangle_nodes_list_temp)) # List of lists of ints. 

    
    def nonlinearization(self):
        """
        Nonlinearize the linear tetrahedral (CST) element to quadratic tetrahedral element.
        """

        self._elem_num = len(self._elem) - 1
        self._orig_node_num = len(self._node) - 1

        self.insertNode()
        self._triangleNodesCollection()

        self._node_num = len(self._node) - 1


def main():
    abaqus_default_directory = "C:/temp" # Default working directory of Abaqus. 
    inp_folder = "inp_files"
    sample_nums = 2500
    data_file_path = "data_head_and_neck.mat"
    node_variable_name, elem_variable_name = "NodeI", "EleI"
    results_folder_path_stress, results_folder_path_coor = "stress", "coor"
    fix_indices_list = [761, 1000, 1158] # Specify the node to fix. At least 3. Indexed from 1. 

    # Generate input file for Abaqus. 
    for i in range(sample_nums):
        if not os.path.isdir(inp_folder): os.mkdir(inp_folder)
        
        file_name_temp = "{}.inp".format(str(i+10001))
        write_path = os.path.join(inp_folder, file_name_temp)

        start_time = time.time()
        inputFile_temp = inputFileGenerator(data_file_path, write_path, fix_indices_list, node_variable_name, elem_variable_name)
        inputFile_temp.writeFile()
        end_time = time.time()
        elapsed_time = end_time - start_time

        print("Input_file: ", file_name_temp, "| Status: Normal | Generation: Completed | Time: %.4f s" % (elapsed_time))
    
    mdict = {"fix_indices_list": fix_indices_list,
             "orig_data_file_name": data_file_path,
             "orig_config_var_name": node_variable_name,
             "inp_folder": inp_folder, 
             "current_directory": os.getcwd(),
             "results_folder_path_stress": results_folder_path_stress,
             "results_folder_path_coor": results_folder_path_coor,
             "original_node_number": inputFile_temp._orig_node_num,
             "loads_num": inputFile_temp.loads_num
            }

    scipy.io.savemat("training_parameters_transfer.mat", mdict)
    
    # np.save(os.path.join(abaqus_default_directory, "training_parameters_transfer.npy"), mdict, fix_imports=True)

    # np.savez(os.path.join(abaqus_default_directory, "training_parameters_transfer.npz"), 
    #          fix_indices_list=fix_indices_list,
    #          orig_data_file_name=data_file_path,
    #          orig_config_var_name=node_variable_name,
    #          inp_folder=inp_folder,
    #          current_directory=os.getcwd(),
    #          results_folder_path_stress=results_folder_path_stress,
    #          results_folder_path_coor=results_folder_path_coor)


if __name__ == "__main__":
    main()
