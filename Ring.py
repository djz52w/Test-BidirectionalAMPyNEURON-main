# Code obtained from
# https://neuron.yale.edu/neuron/docs/ball-and-stick-model-part-4

from neuron import h
from Ball import Ball
from BallAndStick import BallAndStick
import numpy as np

### MPI must be initialized before we create a ParallelContext object
h.nrnmpi_init()
pc = h.ParallelContext()

class Ring:
    """A network of *N* ball-and-stick cells where cell n makes an
    excitatory synapse onto cell n + 1 and the last, Nth cell in the
    network projects to the first cell.
    """
    def __init__(self, N, r, celltype, center, ring_flag, stim_w=0, stim_t=9, stim_delay=1, syn_w=0.01, syn_delay=5):
        """
        :param N: Number of cells.
        :param stim_w: Weight of the stimulus
        :param stim_t: time of the stimulus (in ms)
        :param stim_delay: delay of the stimulus (in ms)
        :param syn_w: Synaptic weight
        :param syn_delay: Delay of the synapse
        :param r: radius of the network
        """
        self._N = N
        self.celltype = celltype
        self.ring_flag = ring_flag
        self.set_gids()                   ### assign gids to processors
        self._syn_w = syn_w
        self._syn_delay = syn_delay
        self._create_cells(r, center)             ### changed to use self._N instead of passing in N
        self._connect_cells()

        ### the 0th cell only exists on one process... that's the only one that gets a netstim
        if stim_w > 0:
            if pc.gid_exists(0):
                self._netstim = h.NetStim()
                self._netstim.number = 1
                self._netstim.start = stim_t
                self._nc = h.NetCon(self._netstim, pc.gid2cell(0).syn)   ### grab cell with gid==0 wherever it exists
                self._nc.delay = stim_delay
                self._nc.weight[0] = stim_w

    def set_gids(self):
        """Set the gidlist on this host."""
        #### Round-robin counting.
        #### Each host has an id from 0 to pc.nhost() - 1.
        self.gidlist = list(range(pc.id(), self._N, pc.nhost()))
        for gid in self.gidlist:
            pc.set_gid2node(gid, pc.id())

    def _create_cells(self, r, center):
        self.cells = {}
        if self.ring_flag:
            offsets = center*np.ones(self._N)
        else:
            offsets = np.array([ [0, 0, 0],
                        [100, 0, 0],
                        [0, 100, 0],
                        [0, 0, 100],
                        [100, 100, 0],
                        [100, 0, 100],
                        [0, 100, 100],
                        [100, 100, 100]])+center

        for i in self.gidlist:    ### only create the cells that exist on this host
            theta = i * 2 * h.PI / self._N + h.PI/4
            if self.ring_flag == 0:
                x = h.cos(theta) * r
                y = h.sin(theta) * r
                z = 0
            else:
                x = 0
                y = 0
                z = 0
            if self.celltype == 'Ball':
                self.cells[i] = Ball(i, x, y, z, theta, offsets[i], self.ring_flag)
            elif self.celltype == 'BallAndStick':
                self.cells[i] = BallAndStick(i, x, y, z, theta, offsets[i], self.ring_flag)

        ### associate the cell with this host and gid
        for ID in self.cells:
            cell = self.cells[ID]
            pc.cell(cell._gid, cell._spike_detector)

    def _connect_cells(self):
        ### this method is different because we now must use ids instead of objects
        for ID in self.cells:
            target = self.cells[ID]
            source_gid = (target._gid - 1 + self._N) % self._N
            nc = pc.gid_connect(source_gid, target.syn)
            nc.weight[0] = self._syn_w
            nc.delay = self._syn_delay
            target._ncs.append(nc)
