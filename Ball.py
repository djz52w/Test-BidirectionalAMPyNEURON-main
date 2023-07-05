# Code bbtained from
# https://neuron.yale.edu/neuron/docs/ball-and-stick-model-part-4

from neuron import h
from Cell import Cell
from neuron.units import ms, mV

class Ball(Cell):
    name = 'Ball'

    def _setup_morphology(self):
        self.soma = h.Section(name='soma', cell=self)
        self.soma.L = self.soma.diam = 12.6157

    def _setup_biophysics(self):
        for sec in self.all:
            sec.insert('extracellular')
            sec.Ra = 100    # Axial resistance in Ohm * cm
            sec.cm = 1      # Membrane capacitance in micro Farads / cm^2

        self.soma.insert('hh')
        for seg in self.soma:
            seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003    # Leak conductance in S/cm2
            seg.hh.el = -54.3     # Reversal potential in mV

        # NEW: the synapse
        self.syn = h.ExpSyn(self.soma(0.5))
        self.syn.tau = 2 * ms
