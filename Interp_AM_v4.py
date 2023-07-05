# Code placed separately from AM.py class in order to use
# numba and jit to increase speed of interpolation

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 21:57:40 2018

@author: guanb
"""
"""
Modified on Thu Nov 14 16:19:38 2019

@author: Jinze
"""
"""
Modified on Mon Mar 02 12:35:47 2021

@author: geneyu
"""

import numpy as np
import math
from numba import jit, types
from numba.typed import Dict, List

@jit(nopython=True)
def Interpolate1D(v1, v2, size, pos):
    if v1 == v2 or size == pos:
        return v2
    else:
        intep_value = ((v2-v1)*(pos/size))+v1
        return intep_value

@jit(nopython=True)
def interp_voltage(node_voltage, Vox_xyz, Vox_size):
    voltage = node_voltage
    
    for ii in range(len(Vox_xyz)):
        x = Vox_xyz[ii][0]
        y = Vox_xyz[ii][1]
        z = Vox_xyz[ii][2]
        
        sx=Vox_size[ii][0]
        sy=Vox_size[ii][1]
        sz=Vox_size[ii][2]
        
        for k in range(z, z+sz+1):
            z1 = Interpolate1D(voltage[x][y][z],voltage[x][y][z+sz],sz,k-z)
            z2 = Interpolate1D(voltage[x][y+sy][z],voltage[x][y+sy][z+sz],sz,k-z)
            z3 = Interpolate1D(voltage[x+sx][y+sy][z],voltage[x+sx][y+sy][z+sz],sz,k-z)
            z4 = Interpolate1D(voltage[x+sx][y][z],voltage[x+sx][y][z+sz],sz,k-z)
            
            for j in range(y, y+sy+1):
                y1 = Interpolate1D(z1,z2,sy,j-y)                                                                                                                                                                                                                                                                                                                                 
                y2 = Interpolate1D(z4,z3,sy,j-y)
                
                for i in range(x, x+sx+1):
                    voltage[i][j][k] = Interpolate1D(y1,y2,sx,i-x)
    
    return voltage

@jit(nopython=True)
def interp_avg_voltage(n1, n2, n3, voltage):
    avg_voltage = np.zeros((n1, n2, n3))
    for k in range(0,n3):
        for j in range(0,n2):
            for i in range(0,n1):
                z1 = Interpolate1D(voltage[i][j][k],voltage[i][j][k+1],1.0,0.5)
                z2 = Interpolate1D(voltage[i][j+1][k],voltage[i][j+1][k+1],1.0,0.5)
                z3 = Interpolate1D(voltage[i+1][j+1][k],voltage[i+1][j+1][k+1],1.0,0.5)
                z4 = Interpolate1D(voltage[i+1][j][k],voltage[i+1][j][k+1],1.0,0.5)
                y1 = Interpolate1D(z1,z2,1.0,0.5)
                y2 = Interpolate1D(z3,z4,1.0,0.5)
                avg_voltage[i][j][k]=Interpolate1D(y1,y2,1.0,0.5)
    
    return avg_voltage

@jit(nopython=True)
def calc_Javg(n1, n2, n3, voltage, voxel_true_size, mat_dict_numba, model_3D, Ex_vox_pre, Ey_vox_pre, Ez_vox_pre, eps0, time_step):
    J_vavg_center = np.zeros((n1, n2, n3))
    for k in range(0,n3):
        for j in range(0,n2):
            for i in range(0,n1):
                Vxyz = voltage[i][j][k]
                Vx1yz = voltage[i+1][j][k]
                Vxy1z = voltage[i][j+1][k]
                Vxyz1 = voltage[i][j][k+1]
                Vx1y1z = voltage[i+1][j+1][k]
                Vx1yz1 = voltage[i+1][j][k+1]
                Vxy1z1 = voltage[i][j+1][k+1]
                Vx1y1z1 = voltage[i+1][j+1][k+1]
                
                Vup = (Vxyz1 + Vx1yz1 + Vx1y1z1 + Vxy1z1)/4
                Vdn = (Vxyz + Vx1yz + Vx1y1z + Vxy1z)/4
                Vlt = (Vxyz + Vx1yz + Vx1yz1 + Vxyz1)/4
                Vrt = (Vxy1z + Vx1y1z + Vx1y1z1 + Vxy1z1)/4
                Vft = (Vx1yz + Vx1y1z + Vx1y1z1 + Vx1yz1)/4
                Vbk = (Vxyz + Vxy1z + Vxy1z1 + Vxyz1)/4    
                
                Vfb = Vft - Vbk
                Vrl = Vrt - Vlt
                Vud = Vup - Vdn
                
                Ex_vox_i = Vfb/(voxel_true_size)
                Ey_vox_i = Vrl/(voxel_true_size)
                Ez_vox_i = Vud/(voxel_true_size)
                
                Jx_vox_center_i=Ex_vox_i/mat_dict_numba[model_3D[k][j][i]][0]+(Ex_vox_i-Ex_vox_pre[i])*mat_dict_numba[model_3D[k][j][i]][1]*eps0/time_step
                Jy_vox_center_i=Ey_vox_i/mat_dict_numba[model_3D[k][j][i]][2]+(Ey_vox_i-Ey_vox_pre[i])*mat_dict_numba[model_3D[k][j][i]][3]*eps0/time_step
                Jz_vox_center_i=Ez_vox_i/mat_dict_numba[model_3D[k][j][i]][4]+(Ez_vox_i-Ez_vox_pre[i])*mat_dict_numba[model_3D[k][j][i]][5]*eps0/time_step
                
                Jdensity_i = math.sqrt(Jx_vox_center_i*Jx_vox_center_i + Jy_vox_center_i*Jy_vox_center_i + Jz_vox_center_i*Jz_vox_center_i)
                J_vavg_center[i][j][k] = Jdensity_i
    
    return J_vavg_center

def Interp_AM(n1, n2, n3, V_n, Vox_xyz, Vox_material, Vox_size, voxel_true_size, mat_dict, time_step, Ex_vox_pre,Ey_vox_pre,Ez_vox_pre,node_voltage,model_3D):
    # node_add_gnd_dict argument removed because it is never used
    # gnd_number never used
    
    eps0=8.854e-12
    
    Vox_size = np.array(Vox_size)
    Vox_xyz = np.array(Vox_xyz)
    #avg_voltage = np.zeros((n1,n2,n3))
    avg_current = np.zeros((n1,n2,n3))
    Vfb_all = list()
    Vrl_all = list()
    Vud_all = list()
    J_vox_center = list()
    I_vox_center = list()
    
    Ex_vox_center = list()
    Ey_vox_center = list()
    Ez_vox_center = list()

    Jdensity_center = list()
    V_vavg_center = np.zeros((n1,n2,n3))
    
    '''
    V_n_gnd=V_n
    

    for i in range (gnd_number):
        V_n_gnd=np.append(V_n_gnd,0)
    '''
    mat_dict_numba = Dict.empty(
        key_type=types.int64,
        value_type=types.float64[:],
    )
    
    for key in mat_dict:
        mat_dict_numba[key] = np.array(mat_dict[key]).astype(np.float64)
    
    voltage = interp_voltage(node_voltage, Vox_xyz, Vox_size)
    avg_voltage = interp_avg_voltage(n1, n2, n3, voltage)
    J_vavg_center = calc_Javg(n1, n2, n3, voltage, voxel_true_size, mat_dict_numba, model_3D, Ex_vox_pre, Ey_vox_pre, Ez_vox_pre, eps0, time_step)
    
    return V_vavg_center, J_vavg_center,voltage, avg_voltage

# End of file