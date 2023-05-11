#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:08:41 2022

@author: jheyston
"""

from scipy.ndimage import gaussian_filter
from numpy import load, save
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm
from math import sqrt
from numpy import linalg as LA
import pandas as pd
import seaborn as sn
from scipy import signal

""
class Gather():
    def __init__(self):
        self.Vx = None
        self.Vy = None
        self.Vz = None
            
class MomentoTensor():
    def __init__(self, Mxx,Myy, Mzz, Mxy, Mxz,  Myz):
        self.xx = Mxx
        self.yy = Myy
        self.zz = Mzz
        self.xy = Mxy
        self.xz = Mxz
        self.yz = Myz
    def __str__(self):
        return f"Mxx: {self.xx:.2f} Mzz: {self.zz:.2f} Mxz: {self.xz:.2f}"


       

class Gradients():
    def __init__(self):
        self.Lambda = np.zeros((ny, nz, nx)).astype(np.float32)
        self.Mu = np.zeros((ny, nz, nx)).astype(np.float32)
        self.Rho = np.zeros((ny, nz, nx)).astype(np.float32)     
    

class Models():
    def __init__(self):
        self.Vp= np.zeros((ny, nz, nx)).astype(np.float32)
        self.Vs = np.zeros((ny, nz, nx)).astype(np.float32)
        self.Rho = np.zeros((ny, nz, nx)).astype(np.float32)      
    

def graficar(model, strName) :
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(model[:, :, corte_x], vmin=-1e-2*np.max(model[:, :, corte_x]), vmax=1e-2*np.max(model[:, :, corte_x]))
    plt.title(strName+' corte X', size=14)
    plt.ylabel('Y (Km)')
    plt.xlabel('Z (Km)')
    plt.colorbar()
        
    plt.subplot(2, 2, 2)
    plt.imshow(model[:, corte_z, :], vmin=-1e-2*np.max(model[:, corte_z, :]), vmax=1e-2*np.max(model[:, corte_z, :]))
    plt.title(strName+' corte Z')
    plt.ylabel('Y (Km)')
    plt.xlabel('X (Km)')
    plt.colorbar()
        
    plt.subplot(2, 2, 3)
    plt.imshow(model[corte_y, :, :], vmin=-1e-2*np.max(model[corte_y, :, :]), vmax=1e-2*np.max(model[corte_y, :, :]))
    plt.title(strName+' corte Y')
    plt.ylabel('Z (Km)')
    plt.xlabel('X (Km)')
    plt.colorbar()



plt.close ('all') 

gk = Gradients()
mk = Models()
cn = Models()

Mobs = MomentoTensor(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    
nx = 150            # Dimension x
ny = 150            # Dimension y
nz = 150            # Dimension z
dx = 1000           # Paso espacial en x
dy = 1000           # Paso espacial en y
dz = 1000           # Paso espacial en z


costFunction = load('../Results/Actual/CostFunction/costFunction.npy')
Mmodxx = load('../Results/Actual/TM/Mmodxx.npy')
Mmodyy = load('../Results/Actual/TM/Mmodyy.npy')
Mmodzz = load('../Results/Actual/TM/Mmodzz.npy')
Mmodxy = load('../Results/Actual/TM/Mmodxy.npy')
Mmodxz = load('../Results/Actual/TM/Mmodxz.npy')
Mmodyz = load('../Results/Actual/TM/Mmodyz.npy')
gk.Rho = load('../Results/Actual/Gradients/gk.Rho.npy')
mk.Rho = load('../Results/Actual/Gradients/mk.Rho.npy')
cn.Rho = load('../Results/Actual/Gradients/cn.Rho.npy')

corte_x = nx//2
corte_y = ny//2
corte_z = nz//2


graficar(mk.Rho, "mk.Rho")
graficar(cn.Rho, "cn.Rho")
graficar(gk.Rho, "gk.Rho")