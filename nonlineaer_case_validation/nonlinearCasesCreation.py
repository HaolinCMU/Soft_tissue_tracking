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

    def __init__(self, data_file_name, write_path, material_type, fix_indices_list, node_variable_name, elem_variable_name):
        """
        Initialize parameters. 

        Parameters: 
        ----------
        data_file_name: String. 
            The file path of information of node, element, etc. 
        write_path: String. 
            The path to write the inp file. 
        material_type: String. 
            The type of material. 
            Used to indicate whether to consider material nonlinearity.
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

        self._modulus = 1e7 # Young's modulus. Unit: Pa. Default: 1e7. 
        self._poisson_ratio = 0.48 # Poisson's ratio. Linear elastic default: 0.3; neo-Hookean default: 0.48.  

        self._isCoupleOn = False # Boolean. True: use coupling constraint; False: do not use coupling constraint. Must not turn on if applying Laplacian smoothing.  
        self._coupling_type = "Kinematic" # String. "Kinematic" / "Distributing". 
        self._coupling_neighbor_layers = 1 # How deep does the neighborhood searching go. Default: 1. 

        self._isLaplacianSmoothingOn = True # Boolean. True: use laplacian smoothing. False: do not use laplacian smoothing.
        self._laplacian_variable_name = "laplacianMatrixI3" 
        self._massMatrix_variable_name = "massMatrixI3" 
        self._laplacian_iter_num = 5
        self._smoothing_rate = 1e-4

        self.loads_num = 3 # For initial testing.
        self._load_sampling_style = "gaussian" # String. Indicating the type of random sampling for force components. "uniform" / "gaussian". 
        self._load_scale = (0.0, 10.0) # Absolute range of the force for uniform sampling. Case and BC specific. (min, max). Unit: N.
        self._gaussian_params = (10.0, 2.0) # Mean and deviation of the force for Gaussian sampling. Case and BC specific. (mean, deviation). Unit: N.
        self._load_params_tuple = None

        self.autoIncrementNum = 5000 # Int. The maximum increment number of the AutoSolver. 
        self.initIncrem = 0.001 # Float. The initial length of the increment (for fixed-step, this is also the length per increm). 
        self.minIncrem = 1e-20 # Float. The minimum increment length for the AutoSolver (ueless for the StaticSolver). 
        self.maxIncrem = 1.0 # Float. The maximum increment length for the AutoSolver (useless for the StaticSovler). 
        self.totalTime = 1.0 # Float. The total time for one simulation step. 
        self.frameNum = 1 # Int. The number of frames intending to extract from the nodal file.

        # ================== Load sampling variables ================== #

        if self._load_sampling_style == "gaussian": self._load_params_tuple = self._gaussian_params
        elif self._load_sampling_style == "uniform": self._load_params_tuple = self._load_scale
        else: 
            self._load_sampling_style = "uniform"
            self._load_params_tuple = self._load_scale

        # ============================================================= #

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

        # Load settings.
        self._loads_nset_name_list = []
        self._rf_name_list = []
        self._rf_nset_name_list = []
        self._rf_nsets = []
        self._load_nsets = [] # Nset definition of loads.
        self._load = self.generateLoadSetting() 

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
        self._loads_posi_indices_list = self._generateLoadPositions(self.loads_num, self._fix_indices_list) # Generate load positions. Randomly. 

        self._surface_list = []
        self._coupling_list = [] 

        self._nset_boundary = [] # All nsets definitions in assembly. Boundary conditions
        self._assembly_end = ["*End Assembly"]
        self._assembly = self.generateAssembly()

        # Material. 
        self.material_type = material_type # String. Indicate material type. "linear"/"neo_hookean_fitting"/"neo_hookean_solid". 
        self._material_def_file_name = "" # Default: "". If there is a file of stress strain definition, please specify here (must not be ""). 
        self._material = self.generateMaterial(self.material_type)
        
        # Boundary condition. 
        self._boundary_initial = ["*Boundary"]
        self._boundary = self.generateBoundaryCondition_fixAll()

        # Step settings.  
        self.freq = int(self.autoIncrementNum / self.frameNum) # Int. The data frame extraction frequency (also refers to the number of increments. Extract one frame per "self.freq" increments). Especially for StaticSolver case.  
        
        self._step = ["*Step, name=step-1, nlgeom=YES, inc={}".format(self.autoIncrementNum),
                      "*Static",
                      "{}, {}, {}, {}".format(self.initIncrem, self.totalTime, 
                                              self.minIncrem, self.maxIncrem)] # Auto solver. 
        self._step_end = ["*End Step"]

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
    

    def writeFile(self, write_status):
        """
        Write 'self.write_lines' into a new inp file. 

        Parameters:
        ----------
            write_status: String. 
                "Normal" / "Fast". 
                    "Normal": generate all definitions; 
                    "Fast": generate nodes and elements definition only. 
        """
        
        if write_status == "Normal":
            self._inputFile_lines_total = (self._header + self._part + self._assembly + 
                                        self._material + self._boundary + self._step + 
                                        self._load + self._resSettings + self._step_end)

            content = '\n'.join(self._inputFile_lines_total)
            
            with open(self.writePath, 'w') as f: f.write(content)
        
        elif write_status == "Fast":
            self._inputFile_lines_total = self._header + self._part

            content = '\n'.join(self._inputFile_lines_total)
            
            with open(self.writePath, 'w') as f: f.write(content)
        
        else:
            self.writeFile("Normal")
    

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
    
    
    def generateMaterial(self, material_type):
        """
        Generate lines for material definition. 

        Parameters:
        ----------
            material_type: String. 
                Indicate what type of material is used. 

        Returns:
        ----------
            material_lines: List of lines. 
                The lines of material definition. 
        """

        material_lines = ["*Material, name={}".format(self._material_name)]

        if material_type == "neo_hookean_fitting":
            stress_strain_lines = self._generateNeoHookeanFitting(self._modulus, (-0.3, 0.3), file_name=self._material_def_file_name)
            material_lines += ["*Hyperelastic, neo hooke, test data input, poisson={}".format(self._poisson_ratio), 
                               "*Uniaxial Test Data"]
            material_lines += stress_strain_lines
        
        elif material_type == "neo_hookean_solid":
            c10 = self._modulus / (4 * (1 + self._poisson_ratio))
            d1 = 6 * (1 - 2 * self._poisson_ratio) / self._modulus
            material_lines += ["*Hyperelastic, neo hooke",
                               "{}, {}".format(c10, d1)]

        elif material_type == "linear":
            material_lines += ["*Elastic",
                               "{}, {}".format(self._modulus, self._poisson_ratio)]
        
        else: material_lines = self.generateMaterial("linear")

        return material_lines


    def _generateNeoHookeanFitting(self, modulus, strain_range, file_name=""):
        """
        Import/Generate stress strain data for neo-Hookean material fitting. 

        Parameters:
        ----------
            modulus: Float. 
                The elastic modulus of material. 
            strain_range: Tuple of floats.
                Range for strain interpolation. 
            file_name (optional): String. 
                The name of stress strain data definition file.
                Default: "".  
        
        Returns:
        ----------
            stress_strain_lines: List of strings. 
                The lines of stress strain data. 
        """

        if file_name != "": return self.readFile(file_name)
        else:
            """
            Assumptions of neo-Hookean formulation:
                Incompressible (Poisson's ratio = ~0.5, small deformation).
                Undergoing uniaxial loading. 
                Formulation: sigma = 2*C*(stretch - 1/(stretch^2)). 
                E = 6*C. 
            """

            strain_data = np.linspace(strain_range[0], strain_range[1], 100)

            stretch_data = strain_data + 1.0
            stress_data = (self._modulus / 3.0) * (stretch_data - 1.0 / stretch_data**2) # Formulation. 

            stress_strain_lines = []

            for i in range(len(stress_data)):
                stress_strain_lines.append("%.6f, %.6f" % (stress_data[i], strain_data[i]))
            
            return stress_strain_lines

    
    def _generateLoadPositions(self, loads_num, fix_indices_list):
        """
        Randomly generate positions of the load. 

        Parameters: 
        ----------
            loads_num: Int. 
                Number of loads.
            fix_indices_list: List of ints. 
                Indices of fixed nodes. 

        Returns:
        ----------
            loads_posi_indices_list: List of ints.
                Picked indices for load application positions. 
        """

        loads_posi_indices_list = []

        for i in range(loads_num):
            while(True):
                load_posi_index_temp = np.random.randint(1, self._surface_nodes_num+1) # Randomly chosen a surface node to apply load F(x, y, z). Indexed from 1. 
                if load_posi_index_temp not in fix_indices_list: break # The randomly generated index cannot be one of the fixed nodes. 
            
            loads_posi_indices_list.append(load_posi_index_temp)

        return loads_posi_indices_list
    

    def _generateLoadValues(self, output_dimension, load_scale, sampling_style="uniform"):
        """
        Randomly generate force values for load component definition.
        Using function: numpy.random.rand(). 

        Parameters:
        ----------
            output_dimension: Tuple of ints.
                The shape of output random array.  
                Size: 2*1. (dim1, dim2). 
            load_scale: Tuple of floats.
                Size: 2*1. (min_laod, max_laod) / (mean, deviation). 
            sampling_style (optional): String. 
                Indicating the type of sampling. 
                    "uniform": uniform distribution. 
                    "gaussian": Gaussian distribution. 
                Default: "uniform". 
        
        Returns:
        ----------
            load_result: Array of floats.
                Size: output_dimension. 
        """
        
        if sampling_style == "uniform":
            load_result = (np.random.rand(output_dimension[0], output_dimension[1]) * 2 - 1) * abs(load_scale[1] - load_scale[0])
            load_result = load_result.reshape(-1,1)

            for index, load_value_temp in enumerate(load_result):
                if load_value_temp < 0: load_result[index] -= self._load_scale[0]
                else: load_result[index] += self._load_scale[0]
            
            load_result = load_result.reshape(output_dimension[0], output_dimension[1])
        
        elif sampling_style == "gaussian":
            mean, deviation = load_scale[0], load_scale[1]
            load_result = np.random.normal(mean, deviation, size=output_dimension)
            load_result = load_result.reshape(-1,1)

            for index, load_value_temp in enumerate(load_result):
                if np.random.rand() <= 0.5: load_result[index] *= -1
            
            load_result = load_result.reshape(output_dimension[0], output_dimension[1])
        
        else: load_result = self._generateLoadValues(output_dimension, load_scale)

        return load_result

    
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
                
        if self._isCoupleOn:
            for i, load_posi_index_temp in enumerate(self._loads_posi_indices_list): 
                ref_name_temp = "rf-{}".format(i+1)
                ref_nset_name_temp = "rf-{}-nset".format(i+1)
                self._rf_name_list.append(ref_name_temp)
                self._rf_nset_name_list.append(ref_nset_name_temp)

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
        
        else:
            if self._isLaplacianSmoothingOn:
                force_vector_temp = np.zeros(shape=(3*self._surface_nodes_num, 1))

                for load_posi_index_temp in self._loads_posi_indices_list:
                    force_vector_temp[(load_posi_index_temp-1)*3:load_posi_index_temp*3,:] = self._generateLoadValues((3,1), self._load_params_tuple, sampling_style=self._load_sampling_style)
                
                laplacian_matrix, mass_matrix = self.data_mat[self._laplacian_variable_name], self.data_mat[self._massMatrix_variable_name]

                force_vector_new = self._laplacianSmoothing(force_vector_temp, laplacian_matrix, mass_matrix, iter_num=self._laplacian_iter_num, 
                                                            smoothing_rate=self._smoothing_rate) # Size: (nSurfI x 3)*1. Fix force value: initial_BC_state="fix". 

                self._loads_posi_indices_list = copy.deepcopy([(list(force_vector_new).index(item)//3)+1 for item in list(force_vector_new) if item != 0]) # Indexed from 1.
                self._loads_posi_indices_list = list(set(self._loads_posi_indices_list))
                self._loads_posi_indices_list.sort()

                for i, load_posi_index_temp in enumerate(self._loads_posi_indices_list):
                    load_nset_name_temp = "Load-{}".format(i+1)
                    self._loads_nset_name_list.append(load_nset_name_temp)
                    
                    self._load_nsets += self.generateNset([load_posi_index_temp], load_nset_name_temp, self._instance_name)
                    self._load = self.generateLoadSetting(force_list=list(force_vector_new.reshape(-1,1)))
            
            else:
                for i, load_posi_index_temp in enumerate(self._loads_posi_indices_list):
                    load_nset_name_temp = "Load-{}".format(i+1)
                    self._loads_nset_name_list.append(load_nset_name_temp)
                    
                    self._load_nsets += self.generateNset([load_posi_index_temp], load_nset_name_temp, self._instance_name)
     
        # Concatenate assembly subparts. 
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
                                    "*{}".format(self._coupling_type)]
    

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
    

    def generateLoadSetting(self, force_list=[]):
        """
        Generate load information. 

        Returns:
        ----------
            load_list: List of strings. 
                Definition of concentrated forces. 
            force_list (optional): List of forces (floats). 
                Size: loads_num * 3. 
                Default: [].
        """

        load_list = []

        if force_list == []: 
            force_list = list(self._generateLoadValues((self.loads_num*3, 1), self._load_params_tuple, sampling_style=self._load_sampling_style))
            
        force_list = np.array(force_list).astype(float).reshape(-1,3) # 2D Array of floats. Size: self._loads_num * 3. 

        if self._isCoupleOn:
            for j, rf_name in enumerate(self._rf_name_list): # Length: self._loads_num
                load_temp = ["*Cload, op=NEW"]

                for i in range(force_list.shape[1]): # 3: Three directions. 
                    load_temp.append("{}, {}, {}".format(rf_name, i+1, force_list[j,i]))
                
                load_list += copy.deepcopy(load_temp)

        else:
            for j, load_name in enumerate(self._loads_nset_name_list): # Length: length of self._loads_nset_name_list. 
                load_temp = ["*Cload"]

                for i in range(force_list.shape[1]): # 3: Three directions.
                    load_temp.append("{}, {}, {}".format(load_name, i+1, force_list[self._loads_posi_indices_list[j]-1,i]))
            
                load_list += copy.deepcopy(load_temp)
       
        return load_list
    
    
    def _laplacianSmoothing(self, force_vector, laplacian_matrix, mass_matrix, iter_num=3, smoothing_rate=1e-4, initial_BC_state=""):
        """
        Implement laplacian smoothing based on pre-calculated Laplacian matrix.
        Formulation: Forward Euler.
            F_(n+1) = (I + lambda*massMatrix*Laplacian) * F_n 

        Parameters:
        ----------
            force_vector: 1D Array of floats.
                With concentrated force values applied at the specidied nodes. 
                Size: (self._surface_nodes_num x 3) * 1. 
            laplacian_matrix: 2D Array of floats.
                Size: (self._surface_nodes_num x 3) * (self._surface_nodes_num x 3).
            mass_matrix: 2D Array of floats.
                Diagonal matrix. 
                Size: (self._surface_nodes_num x 3) * (self._surface_nodes_num x 3).
            Iter_num (optional): Int. 
                The number of smoothing iterations.
                Default: 3. 
            smoothing_rate (optional): float.
                The coefficient that control the step size of smoothing. 
                Default: 1e-4. 
            initial_BC_state (optional): String. 
                Indicating whether to "fix" or "decay" the original concentrated force value.
                Default: "". Indicating smoothing including the original forces.   
        """

        force_vector_new = copy.deepcopy(force_vector)
        for i in range(iter_num): 
            force_vector_new += smoothing_rate * (laplacian_matrix @ force_vector_new) # Without mass matrix. 
            # force_vector_new += smoothing_rate * (mass_matrix @ laplacian_matrix @ force_vector_new) # With mass matrix (NOT recommended). 

            if initial_BC_state == "fix":
                for j, value in enumerate(force_vector):
                    if value != 0:
                        force_vector_new[j] = value

        return force_vector_new


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


def saveLog(file_name_list, elapsed_time_list, write_status, data_file_name, 
            sample_num, fix_indices_list, loads_num, load_sampling_type, load_param_tuple, 
            material_type, modulus, poisson_ratio, isCoupleOn, isLaplacianSmoothingOn, 
            coupling_type="", coupling_neighbor_layer_num=1, 
            laplacian_iter_num=5, laplacian_smoothing_rate=1e-4, write_path="nonlinear_case_generation.log"):
    """
    Save the nonlinear cases generation results into .log file. 

    Parameters:
    ----------
        file_name_list: List of strings. 
            Names of generated files. 
        elapsed_time_list: List of floats. 
            Elapsed time of generation for each input file.
            In exact order. 
        write_status: String. 
            Indicating the type of input file generation. 
            "Normal" / "Fast": 
                "Normal": generate all definitions; 
                "Fast": generate nodes and elements definition only.
        data_file_name: String. 
            The name of modeling data file.
            Format: .mat
        sample_num: Int. 
            Number of generated input files. 
        fix_indices_list: List of ints.
            Indices of fixed points.  
            Indexed from 1. 
        loads_num: Int. 
            The number of concentrated forces.
        load_sampling_type: String.
            The distribution type for force sampling. 
            "uniform" / "gaussian": 
                "uniform": uniform distribution with specified (min, max) range.
                "gaussian": gaussian distribution with specified (mean, dev) parameters. 
        load_param_tuple: tuple of floats.
            Parameters of load sampling. 
            load_sampling_type specific. 
        material_type: String. 
            The type of material.
            "linear" / "neo_hookean_solid" / "neo_hookean_fitting":
                "linear": linear elastic material.
                "neo_hookean_solid": neo-Hookean solid following the stain energy formulation. 
                "neo_hookean_fitting": neo-Hookean solid following the strass-strain curved fitted from user-input strss-strain data. 
        modulus: Float. 
            Elastic modulus of the material.
        poisson_ratio: Float.
            Poisson's ratio of the material. 
        isCoupleOn: Boolean indicator. 
            True: using coupling constraint for local force distribution.
            False: not using coupling constraint.   
        isLaplacianSmoothingOn: Boolean indicator.
            True: using Laplacian-Beltrami operator matrix to smooth the force distribution.
            False: not using Laplacian smoothing. 
        coupling_type (optional): String.
            The type of coupling constraint. 
            Default: "".
        coupling_neighbor_layer_num (optional): Int. 
            The number of neighbor layers to which the local force distributing goes. 
            Default: 1. 
        laplacian_iter_num (optional): Int. 
            The number of iteration for laplacian smoothing. 
            Default: 5.
        laplacian_smoothing_rate (optional): Float. 
            The rate of Laplacian smoothing. 
            Default: 1e-4.
        write_path (optional): String. 
            The path of to-be-written file. 
            Default: "nonlinear_case_generation.log". 
    """

    if isCoupleOn: isCoupleOn_status = "On"
    else: isCoupleOn_status = "Off"

    if isLaplacianSmoothingOn: isLaplacianSmoothingOn_status = "On"
    else: isLaplacianSmoothingOn_status = "Off"

    content = ["Data_file_name: {}".format(data_file_name), 
               "Sample_num = {}".format(sample_num),
               "Fixed_indices_list (indexed from 1): {}".format(fix_indices_list),
               "Material type: {}".format(material_type),
               "Elastic modulus = {} Pa".format(modulus),
               "Poisson's ratio = {}".format(poisson_ratio), 
               "Loads_num = {}".format(loads_num)]

    if load_sampling_type == "uniform":
        content += ["Load sampling type: {}".format(load_sampling_type),
                    "Load sampling range (min, max): {}".format(load_param_tuple)]
    elif load_sampling_type == "gaussian":
        content += ["Load sampling type: {}".format(load_sampling_type),
                    "Load sampling parameters (mean, dev): {}".format(load_param_tuple)]
    else:
        load_sampling_type = "uniform"
        content += ["Load sampling type: {}".format(load_sampling_type),
                    "Load sampling range (min, max): {}".format(load_param_tuple)]

    content += ["Coupling constraint status: {}".format(isCoupleOn_status),
                "Laplacian smoothing status: {}".format(isLaplacianSmoothingOn_status)]

    if isCoupleOn:
        content += ["Coupling type: {}".format(coupling_type),
                    "Coupling neighbor layer numbers: {}".format(coupling_neighbor_layer_num)]
    
    if isLaplacianSmoothingOn:
        content += ["Laplacian smoothing iteration numbers = {}".format(laplacian_iter_num), 
                    "Laplacian smoothing rate = {}".format(laplacian_smoothing_rate)]
    
    content += ["----------------------------------------------------------",
                "Input file\t\tExport status\tGeneration status\tElapsed time/s"]

    elapsed_time_total = 0
    for i, file_name in enumerate(file_name_list):
        data_string_temp = "{}\t\t{}\t\tCompleted\t".format(file_name, write_status) + "\t%.8f" % (elapsed_time_list[i])
        content.append(data_string_temp)
        elapsed_time_total += elapsed_time_list[i]
    
    content += ["----------------------------------------------------------",
                "Total elapsed time: {} s".format(elapsed_time_total)]
    content = '\n'.join(content)
    
    with open(write_path, 'w') as f: f.write(content)


def main():
    abaqus_default_directory = "C:/temp" # Default working directory of Abaqus. 
    inp_folder = "inp_files"
    sample_nums = 2500
    data_file_path = "data_head_and_neck.mat"
    node_variable_name, elem_variable_name = "NodeI", "EleI"
    results_folder_path_stress, results_folder_path_coor = "stress", "coor"
    material_type = "neo_hookean_solid" # "linear" / "neo_hookean_fitting" / "neo_hookean_solid". 
    fix_indices_list = [761, 1000, 1158] # Specify the node to fix. At least 3. Indexed from 1. 
    write_status = "Normal" # String. "Normal" / "Fast". "Normal": generate all definitions; "Fast": generate nodes and elements definition only. 

    # Generate input file for Abaqus. 
    file_name_list, elapsed_time_list = [], []

    for i in range(sample_nums):
        if not os.path.isdir(inp_folder): os.mkdir(inp_folder)
        
        file_name_temp = "{}.inp".format(str(i+10001))
        write_path = os.path.join(inp_folder, file_name_temp)

        start_time = time.time()
        inputFile_temp = inputFileGenerator(data_file_path, write_path, material_type, fix_indices_list, node_variable_name, elem_variable_name)
        inputFile_temp.writeFile(write_status)
        end_time = time.time()
        elapsed_time = end_time - start_time

        file_name_list.append(file_name_temp)
        elapsed_time_list.append(elapsed_time)

        # print(inputFile_temp._loads_posi_indices_list)

        print("Input_file: ", file_name_temp, "| Status:", write_status, "| Generation: Completed | Time: %.4f s" % (elapsed_time))

    saveLog(file_name_list, elapsed_time_list, write_status, data_file_path, sample_nums, 
            fix_indices_list, inputFile_temp.loads_num, inputFile_temp._load_sampling_style, inputFile_temp._load_params_tuple, 
            material_type, inputFile_temp._modulus, inputFile_temp._poisson_ratio, 
            inputFile_temp._isCoupleOn, inputFile_temp._isLaplacianSmoothingOn, 
            coupling_type=inputFile_temp._coupling_type, coupling_neighbor_layer_num=inputFile_temp._coupling_neighbor_layers, 
            laplacian_iter_num=inputFile_temp._laplacian_iter_num, laplacian_smoothing_rate=inputFile_temp._smoothing_rate, 
            write_path="nonlinear_case_generation.log")
    
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
