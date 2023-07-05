#################################################################
# Class file to load and instantiate admittance matrix mesh and #
# extracellular stimulating electrodes.                         #
#################################################################
# @author geneyu

# Original AM code
"""
Created on Mon Dec 24 21:57:40 2018

@author: guanb
"""
"""
Modified on Thu Nov 14 16:19:38 2019

@author: Jinze
"""
"""
Modified on Thu Feb 25 15:21:04 2021

@author: tmillard
"""

import mkl
import os
import gc
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import inspect
from Interp_AM_v4 import *
import pickle

class AM:
    def __init__(self,  precond_method, solve_method, solver_tol, solver_maxiter,
                        netfilename, in_filename, model_2D,
                        n1, n2, n3, time_step, input_type, cur_filenames=None):
        # =============================================================================
        ''' Solver configurations '''
        # =============================================================================
        self.precond_method = precond_method
        self.solver_tol = solver_tol
        self.solve_method = solve_method
        self.solver_maxiter = solver_maxiter

        # =============================================================================
        ''' Input configurations '''
        # =============================================================================
        eps0 = 8.854e-12 # eps0 never used?

        ######################
        # Example file names #
        ######################
        '''
        netfilename = "Sph_80_bipolar.net"
        in_filename = 'Sph_80_bipolar.in'

        # Bipolar stimulation, order same as .net file
        cur_filenames = ["static_1.cur", 'static_2.cur']

        model_2D = np.loadtxt("Sph_80_bipolar.model'
        '''
        self.in_filename = in_filename
        self.netfilename = netfilename

        self.model_2D = model_2D

        ##############n1 n2 n3 are for interpolation
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        model_2D = model_2D.astype('int')
        (r, c) = model_2D.shape
        z_value = n3
        r = int(r / z_value)
        self.model_3D = np.reshape(model_2D, (z_value, r, c))

        ##############Initialize time step value first:
        self.time_step = time_step
        self.total_time_step = 1

        # Solving using stimulating electrodes or NEURON current sources
        # Either 'electrode' or 'ephaptic'
        # Used in self.solve
        self.input_type = input_type

        # =============================================================================
        ''' Initialize data structures '''
        # =============================================================================
        self.parse_infile(in_filename)
        self.parse_netfile(netfilename)

        # =============================================================================
        ''' Building admittance matrix '''
        # =============================================================================
        self.build_admittancematrix()

        # =============================================================================
        ''' Get current amplitudes '''
        # =============================================================================
        if self.input_type == 'electrode':
            self.parse_curfiles(cur_filenames)

        # =============================================================================
        ''' Solve admittance matrix '''
        # =============================================================================
        #self.solve()
        #self.postprocess_solve()

        # =============================================================================
        ''' Interpolate voltages to voxel centers '''
        # =============================================================================
        #V_vavg_center, self.Jvavg_center, node_voltage_interp, self.avg_voltage = Interp_AM(self.n1, self.n2, self.n3, self.V_n, self.Vox_xyz, self.Vox_material,
        #                                                                  self.Vox_size, self.voxel_true_size, self.mat_dict,
        #                                                                  self.time_step,
        #                                                                  self.Ex_vox_pre, self.Ey_vox_pre, self.Ez_vox_pre,
        #                                                                  self.node_voltage, self.model_3D)

    ######################################################################################################

    # =============================================================================
    ''' parse .in file to extract material and true voxel size '''
    # =============================================================================
    def parse_infile(self, in_filename):
        self.voxel_true_size = 0
        self.mat_dict = dict()
        self.Isrc_node = set()
        self.gnd_node = set()
        with open(in_filename) as input_in_file:
            for eachline in input_in_file:
                eachline = eachline.replace("   ", " ")
                eachline = eachline.replace("  ", " ")
                eachline = eachline.replace("    ", " ")
                line_content = eachline.strip('\n').split(' ')
                if line_content[0] == 'unitvoxelsize':
                    self.voxel_true_size = float(line_content[1])

                elif line_content[0] == 'material':
                    self.mat_dict[int(line_content[1])] = ( float(line_content[2]), float(line_content[3]),
                                                            float(line_content[4]), float(line_content[5]),
                                                            float(line_content[6]), float(line_content[7])
                                                            )
                elif line_content[0] == 'spice':
                    self.Isrc_node.add(((int(line_content[4]), int(line_content[5]),
                                        int(line_content[6][:-1])), (int(line_content[8]),
                                        int(line_content[9]), int(line_content[10][:-1])))
                                        )
                elif line_content[0] == 'nodename' and line_content[4] == '0':
                    self.gnd_node.update([( int(line_content[1]), int(line_content[2]),
                                            int(line_content[3]))]
                                            )

    # =============================================================================
    ''' parse .net file '''
    # =============================================================================
    def parse_netfile(self, netfilename):
        node = list()
        self.node_dict = dict()
        self.node_str_dict = dict()
        self.node_str_set = set()

        self.long_gnd_nodename = ''
        self.type_node_value = list()
        self.Vox_xyz = list()
        self.Vox_size = list()
        self.Vox_material = list()

        self.line_num = 0
        finish_file_head = 0
        node_indx = 0

        with open(netfilename) as input_net_file:
            lines = input_net_file.readlines()
            #for eachline in input_net_file:
            for eachline in lines:
                line_parse = eachline.strip('\n').split(' ')
                if len(line_parse) > 1:
                    if finish_file_head == 0:
                        if (line_parse[1] + line_parse[2] == 'paramnodename') and line_parse[-1] == '0':
                            self.long_gnd_nodename = line_parse[-2]
                            ###self.gnd_node.update([(int(line_parse[-2][0:4]), int(line_parse[-2][4:8]), int(line_parse[-2][8:12]))]) # Not used by Emily's code

                        # Not in Emily's script
                        ###if line_parse[1] == "I":
                        ###    self.Isrc_node.add((int(line_parse[-2][0:4]), int(line_parse[-2][4:8]), int(line_parse[-2][8:12])))

                        if line_parse[1] == "voxel":
                            finish_file_head = 1

                            self.Vox_xyz.append((int(line_parse[2]), int(line_parse[3]), int(line_parse[4])))

                            ###### for multi-resolution, voxel size can only be an integer 1,or 2 or 3, correct this to float(line_parse) if it is not the case
                            self.Vox_size.append((int(line_parse[5]), int(line_parse[6]), int(line_parse[7])))
                            self.Vox_material.append(int(line_parse[8]))

                    elif finish_file_head == 1:
                        if line_parse[1] == "voxel":
                            self.Vox_xyz.append((int(line_parse[2]), int(line_parse[3]), int(line_parse[4])))

                            ###### for multi-resolution, voxel size can only be an integer 1,or 2 or 3, correct this to float(line_parse) if it is not the case
                            self.Vox_size.append((int(line_parse[5]), int(line_parse[6]), int(line_parse[7])))
                            self.Vox_material.append(int(line_parse[8]))

                        if line_parse[0] == "Res" or line_parse[0] == "Cap":
                            self.type_node_value.append(line_parse)
                            self.line_num += 1
                            if (line_parse[1] != '0') and ((line_parse[1] in self.node_str_set) == False):
                                self.node_str_set.update([line_parse[1]])
                                self.node_str_dict[line_parse[1]] = node_indx

                                left_node = (int(line_parse[1][0:4]), int(line_parse[1][4:8]), int(line_parse[1][8:12]))
                                node.append(left_node)
                                self.node_dict[left_node] = node_indx
                                node_indx += 1

                            if (line_parse[2] != '0') and ((line_parse[2] in self.node_str_set) == False):
                                self.node_str_set.update([line_parse[2]])
                                self.node_str_dict[line_parse[2]] = node_indx

                                right_node = (int(line_parse[2][0:4]), int(line_parse[2][4:8]), int(line_parse[2][8:12]))
                                node.append(right_node)
                                self.node_dict[right_node] = node_indx
                                node_indx += 1

                    else:
                        print ('error in parsing head of .net file')

        #if len(self.Isrc_node) != len(self.cur_filenames):
        #    raise Exception('inconsistent number of current sources found in .net file and cur_filenames array')

        self.node_no_gnd_number = len(self.node_str_set)

        # Never used
        # node_add_gnd_dict = self.node_dict

        # Never used - maybe for debugging
        self.gnd_number = len(self.gnd_node)

    # =============================================================================
    ''' Building admittance matrix '''
    # =============================================================================
    def build_admittancematrix(self):
        node_pos = 0
        node_neg = 0

        coo_value = []
        coo_row = []
        coo_col = []
        Geq = []
        coo_value_res = []
        coo_row_res = []
        coo_col_res = []

        coo_value_cap = []
        coo_row_cap = []
        coo_col_cap = []
        for i in range(self.line_num):
            if (self.type_node_value[i][1] == "0"):
                node_pos = -1

            else:
                node_pos = self.node_str_dict[self.type_node_value[i][1]]

            if (self.type_node_value[i][2] == "0"):
                node_neg = -1

            else:
                node_neg = self.node_str_dict[self.type_node_value[i][2]]

            if self.type_node_value[i][0] == "Res":
                value = 1 / float(self.type_node_value[i][3])
                if (node_pos != -1 and node_neg != -1):
                    coo_value.append(-value)
                    coo_row.append(node_pos)
                    coo_col.append(node_neg)

                    coo_value.append(-value)
                    coo_row.append(node_neg)
                    coo_col.append(node_pos)

                    coo_value_res.append(-value)
                    coo_row_res.append(node_pos)
                    coo_col_res.append(node_neg)

                    coo_value_res.append(-value)
                    coo_row_res.append(node_neg)
                    coo_col_res.append(node_pos)

                if (node_pos != -1):
                    coo_value.append(value)
                    coo_row.append(node_pos)
                    coo_col.append(node_pos)

                    coo_value_res.append(value)
                    coo_row_res.append(node_pos)
                    coo_col_res.append(node_pos)

                if (node_neg != -1):
                    coo_value.append(value)
                    coo_row.append(node_neg)
                    coo_col.append(node_neg)

                    coo_value_res.append(value)
                    coo_row_res.append(node_neg)
                    coo_col_res.append(node_neg)

            elif self.type_node_value[i][0] == "Cap":
                value = float(self.type_node_value[i][3]) / self.time_step
                if (node_pos != -1 and node_neg != -1):  # if neither node_pos nor node_neg is ground, fill the (node_pos node_neg) of matrix with negative admittance
                    coo_value.append(-value)
                    coo_row.append(node_pos)
                    coo_col.append(node_neg)

                    coo_value.append(-value)
                    coo_row.append(node_neg)
                    coo_col.append(node_pos)

                    coo_value_cap.append(-value)
                    coo_row_cap.append(node_pos)
                    coo_col_cap.append(node_neg)

                    coo_value_cap.append(-value)
                    coo_row_cap.append(node_neg)
                    coo_col_cap.append(node_pos)

                if (node_pos != -1):  # if node_pos is not ground, fill the node_pos at diagonal of Gmatrix with positive admittance
                    coo_value.append(value)
                    coo_row.append(node_pos)
                    coo_col.append(node_pos)

                if (node_neg != -1):
                    coo_value.append(value)
                    coo_row.append(node_neg)
                    coo_col.append(node_neg)

                if (node_pos == -1):  # Geq matrix doesn't need diagonal components, Cap to ground while ==-1
                    coo_value_cap.append(value)
                    coo_row_cap.append(node_neg)
                    coo_col_cap.append(node_neg)

                if (node_neg == -1):
                    coo_value_cap.append(value)
                    coo_row_cap.append(node_pos)
                    coo_col_cap.append(node_pos)

            else:
                print("error in building Gmatirx, can not parse node type, neither Res nor Cap")

        # for the GV=I problem, the negtive I on the right hand side represents the current flowing out of the node
        Gmatrix_coo = sparse.coo_matrix((coo_value, (coo_row, coo_col)), shape=(self.node_no_gnd_number, self.node_no_gnd_number))
        Gmatrix_coo.sum_duplicates()

        ##########  Test purpose
        #
        # G_total_row = coo_row
        # G_total_col = coo_col
        # G_total_value = coo_value
        # print(coo_value)

        self.Gmatrix_csr = Gmatrix_coo.tocsr()

        # Resmatrix_coo and Resmatrix_csr not used at the moment
        #Resmatrix_coo = sparse.coo_matrix((coo_value_res, (coo_row_res, coo_col_res)),
        #                                  shape=(self.node_no_gnd_number, self.node_no_gnd_number))
        #Resmatrix_coo.sum_duplicates()
        #Resmatrix_csr = Resmatrix_coo.tocsr()

        Capmatrix_coo = sparse.coo_matrix((coo_value_cap, (coo_row_cap, coo_col_cap)),
                                          shape=(self.node_no_gnd_number, self.node_no_gnd_number))

        Capmatrix_coo.sum_duplicates()

        self.cap_row = Capmatrix_coo.row
        self.cap_col = Capmatrix_coo.col
        self.cap_val = Capmatrix_coo.data
        del Capmatrix_coo
        gc.collect()

    ########################finish building admittance matrix######################################


    ##############extract location of all sources and set up time domain current source#################
    def parse_curfiles(self, cur_filenames):
        ##########  Test purpose
        # G_total = np.zeros((self.node_no_gnd_number,self.node_no_gnd_number))
        # G_res = np.zeros((self.node_no_gnd_number,self.node_no_gnd_number))
        # Geq = np.zeros((self.node_no_gnd_number,self.node_no_gnd_number))
        #
        # cap_num_1=len(self.cap_val)
        #
        # for i in range(cap_num_1):
        #    Geq[self.cap_row[i]][self.cap_col[i]] = Geq[self.cap_row[i]][self.cap_col[i]] + self.cap_val[i]
        #
        # g_len = len(G_total_value)
        #
        # for i in range(g_len):
        #    G_total[G_total_row[i]][G_total_col[i]] = G_total[G_total_row[i]][G_total_col[i]] + G_total_value[i]

        # =============================================================================
        ''' Read .cur file '''
        # =============================================================================

        with open(cur_filenames[0], "r") as file:
            self.total_time_step = -1
            for line in file:
                if line != "\n":
                    self.total_time_step += 1

        src_number = len(cur_filenames)
        ###self.src_loc = [self.node_dict[x] for x in self.Isrc_node]
        self.src_loc = [[self.node_dict[x[0]], self.node_dict[x[1]]] for x in self.Isrc_node]
        self.Isrc_alltime = np.zeros((self.total_time_step, src_number))

        for i in range(len(cur_filenames)):
            k = 0
            with open(cur_filenames[i]) as input_cur_file:
                for eachline in input_cur_file:
                    line_parse = eachline.strip('\n').split(' ')
                    if line_parse[0] != '%':
                        self.Isrc_alltime[k, i] = float(str(line_parse[0]))
                        k += 1

        #else:
        #    raise Exception('AM expects monopolar or bipolar stimulation')

    def solve(self):
        ###############initialize previous time step E field in the center of voxel##########################
        self.V_n = np.zeros(self.node_no_gnd_number)
        Isrc = np.zeros(self.node_no_gnd_number)
        self.V_static = np.zeros(self.node_no_gnd_number)

        Vox_number = len(self.Vox_size)
        self.Ex_vox_pre = np.zeros(Vox_number)
        self.Ey_vox_pre = np.zeros(Vox_number)
        self.Ez_vox_pre = np.zeros(Vox_number)

        #####################################################################################################
        Niter = 1
        self.Residual = list()

        #
        # =============================================================================
        ''' Initial condition ignored for now '''
        # =============================================================================
        #
        #########  initial condition of voltage given current source, solving for the static case(step = 0) ############################
        # if np.linalg.norm(self.Isrc_alltime[0])!=0:
        #    Isrc[self.src_loc] = self.Isrc_alltime[0]
        #    Isrc_norm=np.linalg.norm(Isrc)
        #
        #    if self.precond_method==1:
        #        precond=sparse.diags(1/Resmatrix_csr.diagonal())
        #    elif self.precond_method==2:
        #        ILU_m=spla.spilu(Resmatrix_csr.tocsc())
        #        Mz= lambda tmp_r: ILU_m.solve(tmp_r)
        #        precond=spla.LinearOperator(Resmatrix_csr.shape, Mz)
        #    else:
        #        print ("error in choosing preconditioner!")
        #
        #    print("finish preconditioning for initial condition solution" )
        #
        #
        #
        #    if self.solve_method==1:
        #        (self.V_n, info)=spla.cg(Resmatrix_csr,Isrc,self.V_n,tol=self.solver_tol,maxiter=self.solver_maxiter,M=precond,callback=callbackfunc, atol=None)
        #    elif self.solve_method==2:
        #        (self.V_n, info)=spla.bicgstab(Resmatrix_csr,Isrc,self.V_n,tol=self.solver_tol,maxiter=self.solver_maxiter,M=precond,callback=None, atol=None)
        #    elif self.solve_method==3:
        #        (self.V_n, info)=spla.bicg(Resmatrix_csr,Isrc,self.V_n,tol=self.solver_tol,maxiter=self.solver_maxiter,M=precond,callback=None, atol=None)
        #    elif self.solve_method==4:
        #        (self.V_n, info)=spla.gmres(Resmatrix_csr,Isrc,self.V_n,tol=self.solver_tol,restart=None, maxiter=self.solver_maxiter,M=precond,callback=None, restrt=None, atol=None)
        #    else:
        #        print("solver is not choosen")
        #    self.V_static = self.V_n
        #
        #
        # end=time.process_time()
        ##plt.plot(Residual)
        #
        # print("finish solving for initial voltage condition in ",end-start, " secs")
        ###########initial condition of voltage given current source calculation done!   #############################

        self.Residual = list()  # reset residual

        #######################with initial condition of self.V_n, calculate all subsequent time steps################
        self.V_record = np.zeros((self.total_time_step, len(self.V_n)))

        #deltav_record = np.zeros((self.total_time_step, len(self.V_n))) #Never used

        if self.precond_method == 1:
            precond = sparse.diags(1 / self.Gmatrix_csr.diagonal())

        elif self.precond_method == 2:
            ILU_m = spla.spilu(self.Gmatrix_csr.tocsc())
            Mz = lambda tmp_r: ILU_m.solve(tmp_r)
            precond = spla.LinearOperator(self.Gmatrix_csr.shape, Mz)

        else:
            print ("error in choosing preconditioner!")

        #print("finish preconditioning for time domain solution")

        # v0 = list() # Never used

        # =============================================================================
        '''Start solving: '''
        # =============================================================================

        #############Here is actually starting from step 1, to n, step 0 is self.V_static##########
        for current_step in range(self.total_time_step):
            Isrc = np.zeros(self.node_no_gnd_number)

            if self.input_type == 'electrode':
                for count, k in enumerate(self.Isrc_node):
                    # Check if monopolar or bipolar stimulation
                    if k[1] in self.gnd_node: # Monopolar
                        Isrc[self.src_loc[count][0]] += self.Isrc_alltime[current_step][count]
                    elif k[0] in self.gnd_node: # Monopolar
                        Isrc[self.src_loc[count][1]] -= self.Isrc_alltime[current_step][count]
                    else: # Bipolar
                        Isrc[self.src_loc[count][0]] += self.Isrc_alltime[current_step][count]
                        Isrc[self.src_loc[count][1]] -= self.Isrc_alltime[current_step][count]

            elif self.input_type == 'ephaptic':
                for ii in range(len(self.src_loc)):
                    Isrc[self.src_loc[ii]] = self.Isrc_alltime[current_step][ii]

            else:
                print('input_type must be electrode or ephaptic!')

            Ieq = np.zeros(self.node_no_gnd_number)

            Ieq = self.calculate_Ieq(self.cap_val, self.cap_row, self.cap_col, Ieq, self.V_n)
            Isrc = Isrc + Ieq
            self.Isrc_norm = np.linalg.norm(Isrc)
            #  Test purpose
            #     print("Isrc[142723]=",Isrc[142723])
            #     print("Ieq[142723]=",Ieq[142723])
            #
            #
            #     print("currentstep=",current_step)
            if (self.Isrc_norm == 0.0):
                self.Isrc_norm = 1

            start_residual = np.linalg.norm(self.Gmatrix_csr * self.V_n - Isrc) / self.Isrc_norm

            if start_residual > self.solver_tol:
                #V_n_pre = self.V_n # Never used
                if self.solve_method == 1:
                    (self.V_n, info) = spla.cg(self.Gmatrix_csr, Isrc, self.V_n, tol=self.solver_tol, maxiter=self.solver_maxiter, M=precond,
                                          callback=self.callbackfunc, atol=None)
                elif self.solve_method == 2:
                    (self.V_n, info) = spla.bicgstab(self.Gmatrix_csr, Isrc, self.V_n, tol=self.solver_tol, maxiter=self.solver_maxiter, M=precond,
                                                callback=self.callbackfunc, atol=None)
                elif self.solve_method == 3:
                    (self.V_n, info) = spla.bicg(self.Gmatrix_csr, Isrc, self.V_n, tol=self.solver_tol, maxiter=self.solver_maxiter, M=precond,
                                            callback=self.callbackfunc, atol=None)
                elif self.solve_method == 4:
                    (self.V_n, info) = spla.gmres(self.Gmatrix_csr, Isrc, self.V_n, tol=self.solver_tol, restart=None, maxiter=self.solver_maxiter,
                                             M=precond, callback=self.callbackfunc, restrt=None, atol=None)
                else:
                    print("solver is not choosen")

            self.V_record[current_step] = self.V_n

    # =============================================================================
    '''Vof post processing for interpolation'''
    # =============================================================================
    def postprocess_solve(self, fname=''):
        self.node_voltage = np.zeros((self.n1 + 1, self.n2 + 1, self.n3 + 1))

        a = list(self.node_str_dict.keys())
        a = np.array(a)
        a = np.insert(a, 0, self.long_gnd_nodename, axis=0)

        V_n_node = np.insert(self.V_n, 0, 0., axis=0)
        V_n_node = np.array(V_n_node)

        for i in range(len(a)):
            x = int(a[i][0:4])
            y = int(a[i][4:8])
            z = int(a[i][8:12])
            self.node_voltage[x][y][z] = V_n_node[i]

        if len(fname) > 0:
            for i in range(self.total_time_step):
                V_n_new = np.insert(self.V_record[i], 0, 0., axis=0)
                V_n_array = np.array(V_n_new)
                filenames = fwrite + "_%d.vof" % i
                with open(filenames, 'wb') as fp:
                    np.savetxt(fp, np.column_stack((a, V_n_array)), fmt="%s")

    # =============================================================================
    '''Interpolate voltages to voxel centers'''
    # =============================================================================
    def interp_voltage(self):
        V_vavg_center, self.Jvavg_center, self.node_voltage_interp, self.avg_voltage = Interp_AM(self.n1, self.n2, self.n3, self.V_n, self.Vox_xyz, self.Vox_material,
                                                                                            self.Vox_size, self.voxel_true_size, self.mat_dict,
                                                                                            self.time_step,
                                                                                            self.Ex_vox_pre, self.Ey_vox_pre, self.Ez_vox_pre,
                                                                                            self.node_voltage, self.model_3D)

    # =============================================================================
    ''' Write out data'''
    # =============================================================================
    def write_data(self, fname):
        vavg_formatted = self.avg_voltage.transpose(2, 1, 0).reshape(-1, self.avg_voltage.shape[0])
        javg_formatted = self.Jvavg_center.transpose(2, 1, 0).reshape(-1, self.Jvavg_center.shape[0])

        with open(fname + '.vavg', 'w') as outfile:
            np.savetxt(outfile, vavg_formatted, delimiter='         ', fmt='%+-10e')

        with open(fname + '.javg', 'w') as outfile:
            np.savetxt(outfile, javg_formatted, delimiter='         ', fmt='%+-10e')

    # =============================================================================

    def calculate_Ieq(self, cap_val, cap_row, cap_col, Ieq, V):
        cap_num = len(cap_val)
        for i in range(cap_num):
            if cap_row[i] == cap_col[i]:
                if (cap_val[i] != 0):
                    Ieq[cap_row[i]] = Ieq[cap_row[i]] - cap_val[i] * V[cap_row[i]]
            else:
                if (cap_val[i] != 0):
                    Ieq[cap_row[i]] = Ieq[cap_row[i]] - cap_val[i] * (V[cap_row[i]] - V[cap_col[i]])

        return Ieq

    def callbackfunc(self, xk):
        frame = inspect.currentframe().f_back
        res = frame.f_locals['resid'] / self.Isrc_norm
        self.Residual.append(res)

# End of file
