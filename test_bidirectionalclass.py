##############################################################
# Example script for using the BidirectionalAMPyNeuron class #
##############################################################
# Neuron instantiation and connectivity obtained from
# https://neuron.yale.edu/neuron/docs/ball-and-stick-model-part-4

from neuron import h
h.nrnmpi_init()
cvode = h.CVode()
cvode.use_fast_imem(1)

pc = h.ParallelContext()
rank = int(pc.id())
nhost = int(pc.nhost())

import numpy as np
from Ring import Ring
import BidirectionalAMPyNeuron

np.random.seed(149)

###########################################
# Initialize NEURON simulation parameters #
###########################################
dt_neuron = 0.1 # ms
tstop = 2 # ms
num_steps = int(tstop/dt_neuron)

#######################################
# Define admittance matrix parameters #
#######################################
precond_method = 1 # ms
solve_method = 1 # Conjugate
#solve_method = 3 # Biconjugate

# solver_tol must be sufficiently small when dealing with few neurons or else
# the solver will return all zeros
solver_tol = 1e-12
solver_maxiter = 3000

# Specify the mesh files
dir_mesh = ''
netfilename = dir_mesh+'Sph_80_bipolar.net'
in_filename = dir_mesh+'Sph_80_bipolar.in'
cur_filenames = ["static_1.cur", 'static_2.cur']  # bipolar stimulation, order same as .net file
model_2D = np.loadtxt(dir_mesh+'Sph_80_bipolar.model')

with open(in_filename, 'r') as f:
    for line in f:
        if line.startswith('world'):
            data = line.split(' ')
            n1 = int(data[1])
            n2 = int(data[2])
            n3 = int(data[3])

dt_AM = 0.1 # ms

#######################
# Instantiate neurons #
#######################
# This network can be replaced with your own network as long as your network
# can be loaded in Python

# Create N number of neurons positioned along a ring of radius r
N = 1 # Number of cells
r = 1 # Radius of circle (microns)
celltype = 'Ball' # or 'BallAndStick'
center = 2000 # Offset of network (microns)
ring_flag = 0 # 0-cube (max 8 neurons, 1-ring (no neuron limit)

ring = Ring(N, r, celltype, center, ring_flag)
cells_dict = ring.cells

###############################################
# Instantiate Bidrectional AM-PyNEURON object #
###############################################
# Must be instantiated after the admittance matrix parameters are specified
# AND
# after the neurons have been instantiated

# Enable ephaptic coupling with 1 or disable with 0
ephaptic_flag = 0

# Specify whether NEURON coordinates are mapped to mesh nodes by
# 'shift': map NEURON to nearest node
# 'split': map NEURON to 8 surrounding nodes and weight by 1/r
mapping_flag = 'shift'

# input_type is sent to the AM class to specify the input type
#  'ephaptic': Solves for both ephaptic coupling and also electrode stimulation
#              if bidirect.load_stimulus is called
# 'electrode': Solves for electrode stimulation using cur_filenames like AM was
#              originally coded for
#              Ex: cur_filenames = ["static_1.cur", 'static_2.cur']
input_type = 'ephaptic'

# Instantiate bidirectional class
fname_save = 'data_testing_mpi.h5'
bidirect = BidirectionalAMPyNeuron.InitBidirectional(
                    cells_dict, ephaptic_flag, mapping_flag,
                    precond_method, solve_method, solver_tol, solver_maxiter,
                    netfilename, in_filename, model_2D,
                    n1, n2, n3, dt_AM, input_type,
                    fname_save, cur_filenames=None)

############################
# Load current stimulation #
############################
# Creating stimulation waveform
# In this case a biphasic pulse
stim_delay = 1  # ms
pulse_width = 1  # ms
stim_waveform = np.zeros(num_steps)
time_array = np.arange(num_steps)*dt_neuron
stim_waveform[time_array >= stim_delay] = -1
stim_waveform[time_array >= stim_delay+pulse_width] = 1
stim_waveform[time_array >= stim_delay+2*pulse_width] = 0

# Load waveform
fname_input = 'data_sphere_bipolar.vavg'

# Tested without extracellular stimulation
bidirect.load_stimulus(fname_input, stim_waveform)

# Testing using current clamp instead
clamps = []
#for ID in cells_dict:
#    clamp = h.IClamp(0.5, sec=cells_dict[ID].soma)
#    clamp.dur = 1
#    clamp.delay = 1
#    clamp.amp = 0.225
#    clamps.append(clamp)

##############################
# Recording neuron variables #
##############################
vsave_res = 0.1 # Sampling frequency of saved data (ms)
e_flag = 1
v_flag = 1
bidirect.record_data(cells_dict, vsave_res, e_flag, v_flag)

######################
# Run the simulation #
######################
save_AM = 1
bidirect.run(dt_neuron, num_steps, save_AM=save_AM, swap1=(0,1))

#############
# Save data #
#############
bidirect.save_data(cells_dict)

pc.barrier()

# End file
a = 1