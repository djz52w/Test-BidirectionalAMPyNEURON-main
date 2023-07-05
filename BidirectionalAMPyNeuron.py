##########################################################################
# Class file to instantiate admittance matrix mesh and run NEURON        #
# simulations with bidirectional communication between admittance matrix #
# and NEURON.                                                            #
##########################################################################
# @author geneyu

from neuron import h
h.nrnmpi_init()
pc = h.ParallelContext()
rank = int(pc.id())
nhost = int(pc.nhost())

import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
import AM
from os import remove
import time as cookie
import h5py

class InitBidirectional:
    def __init__(self,  cells_dict, ephaptic_flag, mapping_flag,
                        precond_method, solve_method, solver_tol, solver_maxiter,
                        netfilename, in_filename, model_2D,
                        n1, n2, n3, dt_AM, input_type,
                        fname_save, cur_filenames=None):

        h.load_file("stdrun.hoc")
        h.load_file("negative_init.hoc")
        self.make_segloc_dict(cells_dict)

        # Instantiate admittance matrix mesh using the AM class
        self.AM_mesh = AM.AM(   precond_method, solve_method, solver_tol, solver_maxiter,
                                netfilename, in_filename, model_2D,
                                n1, n2, n3, dt_AM, input_type, cur_filenames=None)

        self.stim_flag = 0
        self.ephaptic_flag = ephaptic_flag
        self.mapping_flag = mapping_flag
        self.fname_input = ''
        self.fname_save = fname_save
        self.create_datafile()

        # Obtain node coordinates for neurons
        if self.mapping_flag == 'shift':
            self.get_node_coord_shift()
        elif self.mapping_flag == 'split':
            self.get_node_coord_split()
        else:
            print('Error: mapping_flag must be shift or split')

    ##########################################################################
    # Create dictionaries for holding segment locations and segment pointers #
    ##########################################################################
    def make_segloc_dict(self, cells_dict):
        self.segloc_dict = {}
        self.segpointer_dict = {}
        self.num_seg = 0
        for ID in cells_dict:
            self.segloc_dict[ID] = {}
            self.segpointer_dict[ID] = {}
            for sec in cells_dict[ID].all:
                seg_pos = np.linspace(0.5/sec.nseg,1-0.5/sec.nseg,sec.nseg)
                seg_list = [seg for seg in sec]
                self.num_seg += sec.nseg
                for ii in range(len(seg_pos)):
                    for jj in range(int(h.n3d(sec=sec))-1):
                        if (h.arc3d(jj, sec=sec)/sec.L) <= seg_pos[ii] < (h.arc3d(jj+1, sec=sec)/sec.L):
                            swc1 = np.array([h.x3d(jj, sec=sec),h.y3d(jj, sec=sec),h.z3d(jj, sec=sec)])
                            swc2 = np.array([h.x3d(jj+1, sec=sec),h.y3d(jj+1, sec=sec),h.z3d(jj+1, sec=sec)])
                            f = (seg_pos[ii]-h.arc3d(jj, sec=sec)/sec.L)/((h.arc3d(jj+1, sec=sec)-h.arc3d(jj, sec=sec))/sec.L)
                            break

                    self.segloc_dict[ID][str(seg_list[ii])] = f*(swc2-swc1)+swc1
                    self.segpointer_dict[ID][str(seg_list[ii])] = seg_list[ii]

    ###########################################################
    # Load voltage file created by admittance matrix solution #
    ###########################################################
    def load_stimulus(self, fname_input, stim_waveform):
        self.fname_input = fname_input
        dim = [self.AM_mesh.n1, self.AM_mesh.n2, self.AM_mesh.n3]
        with open(fname_input, 'r') as f:
            AM_vavg = f.read()

        AM_vavg = AM_vavg.replace("\n", " ")
        AM_vavg = AM_vavg.replace("  ", " ")
        volt_matrix = AM_vavg.split(" ")
        volt_matrix = list(filter(None, volt_matrix))
        volt_matrix = np.array(list(map(float, volt_matrix)))
        volt_matrix = volt_matrix.reshape(dim[0], dim[1], dim[2])
        self.volt_matrix = volt_matrix * 1000  # turns units from V to mV!!
        self.stim_volts = self.interp_grid2neuron(self.volt_matrix)
        self.stim_waveform = stim_waveform
        self.stim_flag = 1

    ############################################################################
    # Apply the extracellular voltages due to neuron sources and extracellular #
    # stimulating electrode sources.                                           #
    ############################################################################
    # The extracellular voltages are combined linearly.
    def apply_extracellularvoltage(self, ephaptic_volts, time_step, stim_flag=0, ephaptic_flag=0):
       for ID in self.segpointer_dict:
            for ii, seg in enumerate(self.segpointer_dict[ID]):
                # Extracellular potential re-initialized to zero every time due to
                # no memory component (no capacitance). Change if necessary.
                e_extracellular = 0

                # Add extracellular voltage contributions due to neurons
                if ephaptic_flag:
                    e_extracellular += ephaptic_volts[ID][ii]

                # Add extracellular voltage contributions due to stimulating electrodes
                if stim_flag:
                    e_extracellular += self.stim_volts[ID][ii]*self.stim_waveform[time_step]

                self.segpointer_dict[ID][seg].e_extracellular = e_extracellular

    ########################################
    # Obtain node coordinates for neurons #
    ########################################
    # Mapping NEURON compartments to nearest AM mesh node
    def get_node_coord_shift(self):
        # Get valid node indices
        self.node_dict_keys = np.array(list(self.AM_mesh.node_dict.keys()))

        # Get compartment locations to perform interpolation
        node_ind_1ds = []
        self.voxel_size_microns = self.AM_mesh.voxel_true_size/1e-6
        self.coord_compartments = [ [] for ii in range(3) ]
        self.coord_interp_voxel = [ [] for ii in range(3) ]
        self.ID_data_order = []
        self.seg_data_order = []
        for ID in self.segloc_dict:
            for seg in self.segloc_dict[ID]:
                key = str(seg).split('.')
                key = key[1]+'.'+key[2]
                self.ID_data_order.append(ID)
                self.seg_data_order.append(key)

                # Find nearest node 3D coordinate that is
                # also a valid key within self.AM_mesh.node_dict
                node_ind_3d = self.segloc_dict[ID][seg]/self.voxel_size_microns

                d = np.sqrt(((node_ind_3d - self.node_dict_keys)**2).sum(axis=1))

                idx_min = np.argmin(d)
                idx_node = tuple(self.node_dict_keys[idx_min])

                # Obtain compartment coordinates
                self.coord_compartments[0].append(self.segloc_dict[ID][seg][0])
                self.coord_compartments[1].append(self.segloc_dict[ID][seg][1])
                self.coord_compartments[2].append(self.segloc_dict[ID][seg][2])

                # Need to convert into flattened 1D index
                node_ind_1ds.append(self.AM_mesh.node_dict[idx_node])

                # Store actual voxel coordinate
                vox_coord = self.node_dict_keys[idx_min]*self.voxel_size_microns
                self.coord_interp_voxel[0].append(vox_coord[0])
                self.coord_interp_voxel[1].append(vox_coord[1])
                self.coord_interp_voxel[2].append(vox_coord[2])

        pc.barrier()

        # Gather node indices and compartment coordinates across all cores
        self.node_ind_1ds_dict = {}
        for nn in range(nhost):
            buffer = h.Vector(node_ind_1ds)
            pc.broadcast(buffer, nn)
            self.node_ind_1ds_dict[nn] = np.array(buffer).astype(int)

        pc.barrier()

        # Create list of all unique nodes which need to be defined for AM_mesh
        self.src_loc_all = []
        for nn in self.node_ind_1ds_dict:
            self.src_loc_all += list(self.node_ind_1ds_dict[nn])

        self.AM_mesh.src_loc = np.array(list(set(self.src_loc_all)))

    # Find 8 nodes that surround the compartment and their distances/weights
    def get_node_coord_split(self):
        # Get valid node indices
        self.node_dict_keys = np.array(list(self.AM_mesh.node_dict.keys()))

        # Get compartment locations to perform interpolation
        node_ind_1ds = []
        self.voxel_size_microns = self.AM_mesh.voxel_true_size/1e-6
        self.ID_data_order = []
        self.seg_data_order = []
        weights = []
        for ID in self.segloc_dict:
            for seg in self.segloc_dict[ID]:
                key = str(seg).split('.')
                key = key[1]+'.'+key[2]
                self.ID_data_order.append(ID)
                self.seg_data_order.append(key)

                # Find nearest bounding 3D node coordinates that are
                # also valid keys within self.AM_mesh.node_dict
                node_ind_3d = self.segloc_dict[ID][seg]/self.voxel_size_microns

                d = np.zeros(self.node_dict_keys.shape)
                for ii in range(len(node_ind_3d)):
                    d[:, ii] = node_ind_3d[ii] - self.node_dict_keys[:, ii]

                idx_bounds = []
                idx_bounds.append(np.all([d[:, 0] >= 0, d[:, 1] >= 0, d[:, 2] >= 0], axis=0))
                idx_bounds.append(np.all([d[:, 0] < 0, d[:, 1] >= 0, d[:, 2] >= 0], axis=0))
                idx_bounds.append(np.all([d[:, 0] >= 0, d[:, 1] < 0, d[:, 2] >= 0], axis=0))
                idx_bounds.append(np.all([d[:, 0] >= 0, d[:, 1] >= 0, d[:, 2] < 0], axis=0))
                idx_bounds.append(np.all([d[:, 0] < 0, d[:, 1] < 0, d[:, 2] >= 0], axis=0))
                idx_bounds.append(np.all([d[:, 0] < 0, d[:, 1] >= 0, d[:, 2] < 0], axis=0))
                idx_bounds.append(np.all([d[:, 0] >= 0, d[:, 1] < 0, d[:, 2] < 0], axis=0))
                idx_bounds.append(np.all([d[:, 0] < 0, d[:, 1] < 0, d[:, 2] < 0], axis=0))

                bounds = []
                sigmas = []
                for ii in range(len(idx_bounds)):
                    if d[idx_bounds[ii]].size > 0:
                        r = np.sqrt((d[idx_bounds[ii]]**2).sum(axis=1))
                        idx_nearest = np.argmin(r)
                        bounds.append(tuple(self.node_dict_keys[idx_bounds[ii]][idx_nearest]))
                        sigmas.append(1/r.min())
                    else:
                        print('Error: Compartment coordinate not within mesh!!!')

                # Compute the proportion of current that will go to the node
                sigma_total = np.sum(sigmas)
                for ii in range(len(sigmas)):
                    weights.append(sigmas[ii]/sigma_total)

                # Need to convert into the flattened 1D index that the AM_mesh
                # uses to reference its node list
                for idx_node in bounds:
                    node_ind_1ds.append(self.AM_mesh.node_dict[idx_node])

        pc.barrier()

        # Gather node indices across all cores
        node_ind_1ds_dict = {}
        for nn in range(nhost):
            buffer = h.Vector(node_ind_1ds)
            pc.broadcast(buffer, nn)
            node_ind_1ds_dict[nn] = np.array(buffer).astype(int)

        weights_dict = {}
        for nn in range(nhost):
            buffer = h.Vector(weights)
            pc.broadcast(buffer, nn)
            weights_dict[nn] = np.array(buffer)

        pc.barrier()

        # Create list of all unique nodes which need to be defined for AM_mesh
        self.src_loc_all = []
        self.weights_all = []
        for nn in range(nhost):
            self.src_loc_all += list(node_ind_1ds_dict[nn])
            self.weights_all += list(weights_dict[nn])

        self.AM_mesh.src_loc = np.array(list(set(self.src_loc_all)))

    #########################################################################
    # Using scipy's RegularGridInterpolator to calculate interpolation from #
    # admittance mesh coordinates to NEURON compartment coordinates         #
    #########################################################################
    # Mapping AM node to NEURON compartments
    def interp_grid2neuron(self, voltage_mat):
        # There needs to be a 0.5 node index offset because voltage_mat
        # corresponds to the voltages at the voxel centers.
        coord_x = self.voxel_size_microns*(np.arange(self.AM_mesh.n1)+0.5)
        coord_y = self.voxel_size_microns*(np.arange(self.AM_mesh.n2)+0.5)
        coord_z = self.voxel_size_microns*(np.arange(self.AM_mesh.n3)+0.5)

        func_interp = RegularGridInterpolator((coord_x, coord_y, coord_z), voltage_mat)
        cells_volts = {}
        for ID in self.segloc_dict:
            locs_dict = self.segloc_dict[ID]
            all_pts = np.array(list(locs_dict.values()))
            cells_volts[ID] = func_interp(all_pts)

        return cells_volts

    ############################################
    # Record data from all NEURON compartments #
    ############################################
    # Uses same data structure as segment locations
    def record_data(self, cells_dict, res, e_flag, v_flag):
        self.res_data = res
        self.data = {}
        if e_flag:
            self.data['e extracellular'] = {}

        if v_flag:
            self.data['membrane voltage'] = {}

        self.data['membrane current'] = {}
        for ID in cells_dict:
            if e_flag:
                self.data['e extracellular'][ID] = {}

            if v_flag:
                self.data['membrane voltage'][ID] = {}

            self.data['membrane current'][ID] = {}
            for sec in cells_dict[ID].all:
                for seg in sec:
                    key = str(seg).split('.')
                    key = key[1]+'.'+key[2]

                    if e_flag:
                        self.data['e extracellular'][ID][key] = h.Vector()
                        self.data['e extracellular'][ID][key].record(seg._ref_e_extracellular, res, sec=sec)

                    if v_flag:
                        self.data['membrane voltage'][ID][key] = h.Vector()
                        self.data['membrane voltage'][ID][key].record(seg._ref_v, res, sec=sec)

                    self.data['membrane current'][ID][key] = h.Vector()
                    self.data['membrane current'][ID][key].record(seg._ref_i_membrane_, res, sec=sec)

    #################################################################
    # Run function for enabling communication between NEURON and AM #
    #################################################################
    def run(self, dt_neuron, num_steps, celsius=None, save_AM=0, flip1=None, flip2=None, flip3=None,
            swap1=None, swap2=None):
        h.dt = dt_neuron
        pc.set_maxstep(10.0)
        if celsius != None:
            h.celsius = celsius

        h.negative_init()
        tfin = 0
        if rank == 0:
            ST = cookie.time()
            print("Initialization Complete, Simulation Starting...")

        self.save_AM = save_AM
        if save_AM:
            if rank == 0:
                with h5py.File(self.fname_save, 'w') as dset:
                    dset.create_dataset('AM voltage', (num_steps, self.AM_mesh.n1, self.AM_mesh.n2, self.AM_mesh.n3))

        pc.barrier()

        ephaptic_volts = np.zeros((self.AM_mesh.n1, self.AM_mesh.n2, self.AM_mesh.n3))
        for step in range(num_steps):
            # Apply extracellular voltages as stimulation to neurons
            self.apply_extracellularvoltage(ephaptic_volts, step, self.stim_flag, self.ephaptic_flag)

            #########################
            # Run NEURON simulation #
            #########################
            tfin += dt_neuron

            pc.barrier()
            h.continuerun(tfin)

            ####################################
            # Calculate extracellular voltages #
            ####################################
            # If the current time is a multiple of the AM time step, then update
            # the ephaptic voltages.
            if (tfin % self.AM_mesh.time_step) < self.AM_mesh.time_step:
                # Get all currents in same order as self.coord_nrn
                self.current = np.zeros(self.num_seg)

                # Assume all currents are zero at first time step of simulation
                if step > 0:
                    for ii in range(self.num_seg):
                        ID = self.ID_data_order[ii]
                        seg = self.seg_data_order[ii]
                        self.current[ii] = self.data['membrane current'][ID][seg][-1]

                pc.barrier()

                # Broadcast currents to all cores
                current_all = []
                for nn in range(nhost):
                    buffer = h.Vector(self.current)
                    pc.broadcast(buffer, nn)
                    current_all += list(buffer)

                # Construct Isrc_alltime vector and combine currents where nearest node index overlaps
                self.AM_mesh.Isrc_alltime = np.zeros((1, len(self.AM_mesh.src_loc)))

                # Need to add the nrn currents to the correct AM node indices
                map_idx_unique = np.zeros(len(self.src_loc_all)).astype(int)
                for ii in range(len(map_idx_unique)):
                    map_idx_unique[ii] = np.where(self.AM_mesh.src_loc == self.src_loc_all[ii])[0][0]

                for ii in range(len(map_idx_unique)):
                    idx = map_idx_unique[ii]
                    if self.mapping_flag == 'shift':
                        self.AM_mesh.Isrc_alltime[0][idx] += current_all[ii]
                    elif self.mapping_flag == 'split':
                        ind_current = ii // 8
                        self.AM_mesh.Isrc_alltime[0][idx] += self.weights_all[ii]*current_all[ind_current]

                # Convert from nA to A
                self.AM_mesh.Isrc_alltime[0] *= 1e-9

                # Solve the admittance matrix
                self.AM_mesh.solve()
                self.AM_mesh.postprocess_solve()
                self.AM_mesh.interp_voltage()

                # Convert from V to mV
                self.AM_mesh.avg_voltage *= 1000

                # Flipping
                if swap1 != None:
                    self.AM_mesh.avg_voltage = np.swapaxes(self.AM_mesh.avg_voltage, swap1[0], swap1[1])

                if swap2 != None:
                    self.AM_mesh.avg_voltage = np.swapaxes(self.AM_mesh.avg_voltage, swap2[0], swap2[1])

                if flip1 != None:
                    self.AM_mesh.avg_voltage = np.flip(self.AM_mesh.avg_voltage, flip1)

                if flip2 != None:
                    self.AM_mesh.avg_voltage = np.flip(self.AM_mesh.avg_voltage, flip2)

                if flip3 != None:
                    self.AM_mesh.avg_voltage = np.flip(self.AM_mesh.avg_voltage, flip3)

                # Interpolate AM voltages onto neuron compartments
                ephaptic_volts = self.interp_grid2neuron(self.AM_mesh.avg_voltage)

            # Save AM voltages if flag is set
            if save_AM:
                if rank == 0:
                    if step < (num_steps-1):
                        with h5py.File(self.fname_save, 'a') as dset:
                            dset['AM voltage'][step+1] = self.AM_mesh.avg_voltage

        if rank == 0:
            ET = cookie.time() - ST
            print("Wall-time (simulation only) = %.3f seconds." % ET)

    ######################
    # Write data to file #
    ######################
    def create_datafile(self):
        if rank == 0:
            with h5py.File(self.fname_save, 'w') as dset:
                pass

    def save_data(self, cells_dict, num_tri=5):
        if rank == 0:
            with h5py.File(self.fname_save, 'a') as dset:
                for var in self.data:
                    dset.create_group(var)
                    for ID in self.data[var]:
                        dset[var].create_group(str(ID))
                        for seg in self.data[var][ID]:
                            dset[var][str(ID)][seg] = np.array(self.data[var][ID][seg])

                dset.create_group('seg order')
                offset = 0
                x_all = []
                y_all = []
                z_all = []
                ID_all = []
                connections_all = []
                x_compartment = []
                y_compartment = []
                z_compartment = []
                dset.create_group('nrn compartment locs')
                for ID in cells_dict:
                    endpoints, radii, seg_order = cells_dict[ID]._get_data()
                    ascii_list = [seg.encode("ascii", "ignore") for seg in seg_order]
                    dset['seg order'][str(ID)] = ascii_list
                    x, y, z, connections = cells_dict[ID]._connect_dots(endpoints, radii, num_tri, offset)
                    x_all.append(x)
                    y_all.append(y)
                    z_all.append(z)
                    ID_all.append(ID*np.ones(len(z), dtype=int))
                    connections_all += connections
                    offset += len(z)

                    # Save compartment locations
                    dset['nrn compartment locs'].create_group(str(int(ID)))
                    sec = list(self.segpointer_dict[ID].keys())[0].split('.')[0]
                    for ii in range(len(seg_order)):
                        seg = seg_order[ii]
                        dset['nrn compartment locs'][str(int(ID))][seg] = self.segloc_dict[ID][sec+'.'+seg]

                x_all = np.hstack(x_all)
                y_all = np.hstack(y_all)
                z_all = np.hstack(z_all)
                ID_all = np.hstack(ID_all)
                dset.create_group('cylinder plot data')
                dset['cylinder plot data'][str(rank)] = np.vstack([x_all, y_all, z_all, ID_all])
                dset.create_group('cylinder connections')
                dset['cylinder connections'][str(rank)] = connections_all

                if self.stim_flag:
                    dset['stimulus waveform'] = self.stim_waveform
                    dset.attrs['stimulus input file'] = self.fname_input

                dset.attrs['dt neuron'] = h.dt
                dset.attrs['dt AM'] = self.AM_mesh.time_step
                dset.attrs['dt data'] = self.res_data
                dset.attrs['mesh dimensions'] = [self.AM_mesh.n1, self.AM_mesh.n2, self.AM_mesh.n3]
                dset.attrs['voxel resolution microns'] = self.voxel_size_microns
                dset.attrs['num tri'] = num_tri

                #if self.save_AM:
                #    dset['AM voltage'] = self.meshv_ephaptic

        for ii in range(1, nhost):
            pc.barrier()
            if rank == ii:
                if len(cells_dict) > 0:
                    with h5py.File(self.fname_save, 'a') as dset:
                        for var in self.data:
                            for ID in self.data[var]:
                                dset[var].create_group(str(ID))
                                for seg in self.data[var][ID]:
                                    dset[var][str(ID)][seg] = np.array(self.data[var][ID][seg])

                        offset = 0
                        x_all = []
                        y_all = []
                        z_all = []
                        ID_all = []
                        connections_all = []
                        for ID in cells_dict:
                            endpoints, radii, seg_order = cells_dict[ID]._get_data()
                            ascii_list = [seg.encode("ascii", "ignore") for seg in seg_order]
                            dset['seg order'][str(ID)] = ascii_list
                            x, y, z, connections = cells_dict[ID]._connect_dots(endpoints, radii, num_tri, offset)
                            x_all.append(x)
                            y_all.append(y)
                            z_all.append(z)
                            ID_all.append(ID*np.ones(len(z), dtype=int))
                            connections_all += connections
                            offset += len(z)

                            # Save compartment locations
                            dset['nrn compartment locs'].create_group(str(int(ID)))
                            for ii in range(len(seg_order)):
                                seg = seg_order[ii]
                                start_point, end_point = np.array(endpoints[ii])
                                mid_point = 0.5*(start_point+end_point)
                                dset['nrn compartment locs'][str(int(ID))][seg] = mid_point

                        x_all = np.hstack(x_all)
                        y_all = np.hstack(y_all)
                        z_all = np.hstack(z_all)
                        ID_all = np.hstack(ID_all)

                        dset['cylinder plot data'][str(rank)] = np.vstack([x_all, y_all, z_all, ID_all])
                        dset['cylinder connections'][str(rank)] = connections_all

# End file
