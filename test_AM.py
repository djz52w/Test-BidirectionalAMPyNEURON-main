###########################################################################
# Example code for using AM class to simulate the propagation of electric #
# field given a mesh and a bipolar electrode configuration                #   
###########################################################################

from AM import AM
import numpy as np

#######################################
# Define admittance matrix parameters #
#######################################
precond_method = 1
solve_method = 1

solver_tol = 1e-4
solver_maxiter = 3000

eps0 = 8.854e-12
netfilename = "Sph_80_bipolar.net"
in_filename = 'Sph_80_bipolar.in'
cur_filenames = ["static_1.cur", 'static_2.cur']  # bipolar stimulation, order same as .net file
model_2D = np.loadtxt("Sph_80_bipolar.model")

print("Simulation file: ", netfilename)

##############n1 n2 n3 are for interpolation
n1 = 80
n2 = 80
n3 = 80

# Get time step
with open(cur_filenames[0]) as input_cur_file:
    for eachline in input_cur_file:
        line_parse = eachline.strip('\n').split(' ')
        if line_parse[0] == '%':
            time_step = line_parse[1]

time_step = float(time_step)

AM_sphere = AM( precond_method, solve_method, solver_tol, solver_maxiter, 
                netfilename, in_filename, model_2D,
                n1, n2, n3, time_step, 'electrode', cur_filenames)

##########################
# Get current amplitudes #
##########################
AM_sphere.parse_curfiles(cur_filenames)

###########################
# Solve admittance matrix #
###########################
AM_sphere.solve()
AM_sphere.postprocess_solve()

#########################################
# Interpolate voltages to voxel centers #
#########################################
AM_sphere.interp_voltage()

##################
# Write voltages #
##################
AM_sphere.write_data('data_sphere_bipolar')