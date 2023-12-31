######################################################################
# Plot 3D data generated from a bidirectional AM PyNeuron simulation #
######################################################################
# @author geneyu
import h5py
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt

#############
# Load data #
#############
# Name of data file generated by simulation
fname_data = 'data_testing_mpi.h5'
data = {}
with h5py.File(fname_data, 'r') as dset:
    vars = ['membrane voltage', 'membrane current', 'e extracellular']
    for var in vars:
        if var in dset:
            data[var] = {}
            for ID in dset[var]:
                data[var][int(ID)] = {}
                for seg in dset[var][ID]:
                    data[var][int(ID)][seg] = dset[var][ID][seg][:]
    
    seg_order = {}
    for ID in dset['seg order']:
        seg_order[int(ID)] = dset['seg order'][ID][:].astype(str)
    
    plot_data = []
    connections = []
    offset = 0
    for rank in dset['cylinder plot data']:
        plot_data.append(dset['cylinder plot data'][rank][:])
        connections.append(dset['cylinder connections'][rank][:].T)
        connections[-1] += offset
        offset += connections[-1].shape[1]
    
    x, y, z, IDs = np.hstack(plot_data) # microns
    connections = np.hstack(connections).T
    
    ephaptic_volt = dset['AM voltage'][:] # mV
    
    stim_waveform = dset['stimulus waveform'][:]
    time = dset.attrs['dt neuron']*np.arange(stim_waveform.size) # ms
    fname_input = dset.attrs['stimulus input file']
    dim = dset.attrs['mesh dimensions']
    res = dset.attrs['voxel resolution microns']
    num_tri = dset.attrs['num tri']

with open(fname_input, 'r') as f:
    stim_volt = f.read()

stim_volt = stim_volt.replace("\n", " ")
stim_volt = stim_volt.replace("  ", " ")
stim_volt = stim_volt.split(" ")
stim_volt = list(filter(None, stim_volt))
stim_volt = np.array(list(map(float, stim_volt)))
stim_volt = stim_volt.reshape((dim[0], dim[1], dim[2]))
stim_volt = stim_volt * 1000 # Convert V to mV

################
# Create plots #
################
x_vox = np.arange(dim[0])*res
y_vox = np.arange(dim[1])*res
z_vox = np.arange(dim[2])*res
X, Y, Z = np.meshgrid(x_vox, y_vox, z_vox)

# Specify which time of the simulation is plotted
time_plot = 1.1# ms
print(time)
t_idx = np.where(time == time_plot)[0][0]
print(t_idx)
# Reconstruct extracellular voltage
v_extracellular = stim_volt*stim_waveform[t_idx] + ephaptic_volt[t_idx]

# Set data thresholds for plotting
# Values below threshold will not be plotted
v_thresh = np.abs(v_extracellular).mean()*25
idxv = np.abs(v_extracellular) > v_thresh

# Limits for colormap
vlim = 0.15*np.max(np.abs(v_extracellular))
print("vlim: ", vlim)
# Reconstruct NEURON data for plotting
var_plot = data['e extracellular']
data_plot = np.zeros(len(x))
for ID in data['membrane current']:
    idx = np.arange(len(x))[IDs == ID]
    for ii in range(len(seg_order[ID])):
        ind0 = int(2*ii*num_tri)
        ind1 = int((2*ii+1)*num_tri)
        ind2 = int((2*ii+2)*num_tri)
        data_plot[idx[ind0]:idx[ind1]] = var_plot[ID][seg_order[ID][ii]][t_idx]
        if ii < (len(seg_order[ID])-1):
            data_plot[idx[ind1]:idx[ind2]] = var_plot[ID][seg_order[ID][ii]][t_idx]
        else:
            data_plot[idx[ind1]:] = var_plot[ID][seg_order[ID][ii]][t_idx]

# Generate plot
f = mlab.figure(bgcolor=(0.5,0.5,0.5),fgcolor=(0.5,0.5,0.5))
f.scene.renderer.use_depth_peeling=True

# Plotting neurons
g1 = mlab.triangular_mesh(  x, y, z, connections, scalars=data_plot,
                            colormap='coolwarm', opacity=1.0,
                            vmin=-vlim, vmax=vlim
                            )
g1.actor.property.frontface_culling = True

#Plotting extracellular voltage
g2 = mlab.points3d( X[idxv], Y[idxv], Z[idxv], v_extracellular[idxv],
                    colormap='coolwarm', opacity=0.01,
                    vmin=-vlim, vmax=vlim
                    )
mlab.show()
#mlab.savefig('myfigure.png')
# End file