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


            
    
    

plt.close ('all') 
path = 'Actual'
M = np.load(f'../Results/{path}/TM/targetTM.npy')
Mobs = MomentoTensor(M[0], M[1],M[2] , M[3], M[4], M[5])

    
''' Geometry 3D'''
rec_pos_x = load('../Results/Actual/Geometry/sta_pos_x.npy')
rec_pos_y = load('../Results/Actual/Geometry/sta_pos_y.npy')
rec_pos_z = load('../Results/Actual/Geometry/sta_pos_z.npy')

src_pos_x = load('../Results/Actual/Geometry/sismos_pos_x.npy')
src_pos_y = load('../Results/Actual/Geometry/sismos_pos_y.npy')
src_pos_z = load('../Results/Actual/Geometry/sismos_pos_z.npy')


fig = plt.figure(figsize=(20,10))
ax = plt.axes(projection='3d')
ax.scatter3D(rec_pos_x, rec_pos_y, -rec_pos_z,  cmap='Greens', label='Estaciones');
ax.scatter3D(src_pos_x, src_pos_y, -src_pos_z,  cmap='Greens', label='Sismos');
ax.set_xlabel('X', size=18)
ax.set_ylabel('Y', size=18)
ax.set_zlabel('Z', size=18)
ax.legend()
ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(-nz, 0))
fig.savefig("../pdf/Actual/geometry3D.pdf", bbox_inches='tight')


obsData = Gather()
modData = Gather()
modData0 = Gather()
resData = Gather()


modData0.Vx = load('../Results/Actual/Gathers/modData0.Vx.npy')
modData0.Vy = load('../Results/Actual/Gathers/modData0.Vy.npy')
modData0.Vz= load('../Results/Actual/Gathers/modData0.Vz.npy')

modData.Vx = load('../Results/Actual/Gathers/modData.Vx.npy')
modData.Vy = load('../Results/Actual/Gathers/modData.Vy.npy')
modData.Vz= load('../Results/Actual/Gathers/modData.Vz.npy')

obsData.Vx = load('../Results/Actual/Gathers/obsData0.Vx.npy')
obsData.Vy = load('../Results/Actual/Gathers/obsData0.Vy.npy')
obsData.Vz= load('../Results/Actual/Gathers/obsData0.Vz.npy')

resData.Vx = load('../Results/Actual/Gathers/resData.Vx.npy')
resData.Vy = load('../Results/Actual/Gathers/resData.Vy.npy')
resData.Vz= load('../Results/Actual/Gathers/resData.Vz.npy')

###############################################################################
f = plt.figure(figsize=(20,10))
plt. subplot(2,3,1)
plt.plot(obsData.Vx[:,0], label='obsData.Vx')
plt.plot(modData.Vx[:,0], label='modData.Vx ite 200')
plt.plot(modData0.Vx[:,0], label='modData.Vx ite 1')
plt.xticks([])
plt.title('station 1')
plt.legend(loc='lower right')

plt.ylabel('Amplitud', size=18)
plt. subplot(2,3,4)
plt.plot(obsData.Vx[:,1], label='obsData.Vx')
plt.plot(modData.Vx[:,1], label='modData.Vx ite 200')
plt.plot(modData0.Vx[:,1], label='modData.Vx ite 1')
plt.xticks([])
plt.title('station 2')
plt.ylabel('Amplitud', size=18)
plt.legend(loc='lower right')


###############################################################################


plt. subplot(2,3,2)
plt.plot(obsData.Vy[:,0], label='obsData.Vy')
plt.plot(modData.Vy[:,0], label='modData.Vy ite 200')
plt.plot(modData0.Vy[:,0], label='modData.Vy ite 1')
plt.xticks([])
plt.title('station 1')
plt.legend(loc='lower right')

plt. subplot(2,3,5)
plt.plot(obsData.Vy[:,1], label='obsData.Vy')
plt.plot(modData.Vy[:,1], label='modData.Vy ite 200')
plt.plot(modData0.Vy[:,1], label='modData.Vy ite 1')
plt.xticks([])
plt.title('station 2')
plt.legend(loc='lower right')


###############################################################################
plt. subplot(2,3,3)
plt.plot(obsData.Vz[:,0], label='obsData.Vz')
plt.plot(modData.Vz[:,0], label='modData.Vz ite 200')
plt.plot(modData0.Vz[:,0], label='modData.Vz ite 1')
plt.title('station 1')
plt.xticks([])
plt.legend(loc='lower right')

plt. subplot(2,3,6)
plt.plot(obsData.Vz[:,1], label='obsData.Vz')
plt.plot(modData.Vz[:,1], label='modData.Vz ite 200')
plt.plot(modData0.Vz[:,1], label='modData.Vz ite 1')
plt.xticks([])
plt.title('station 2')
plt.legend(loc='lower right')
f.savefig("../pdf/Actual/datos_obs_mod.pdf", bbox_inches='tight')


#data = np.array([np.ravel(obsData.Vx), np.ravel(modData.Vx)]).T
#nt, stations = obsData.Vx.shape
#
#corr1 = signal.correlate2d(obsData.Vx, modData.Vx, boundary='symm', mode='same')
#
#plt.figure()
#plt.imshow(corr1, aspect='auto')
#plt.show()

#x = obsData.Vx[:,1]
#y = modData.Vx[:,1]
#R1 = np.corrcoef(x, y, rowvar = False)
#R2 = np.corrcoef(x, x, rowvar = False)
#
#
#l2_dobsVx = np.sqrt( np.sum( x * x ))
#l2_dmodVx = np.sqrt( np.sum( y * y ))
#dmod_dobs_Vx = np.sum(  x *  y)
#print(np.round((dmod_dobs_Vx) / (l2_dmodVx * l2_dobsVx),decimals=8) )
#
#np.corrcoef(x[:,0], y[:,1], rowvar=False)
#CovXY = np.cov([x,y],  ddof=0)
#CovX = np.cov(x,x,  ddof=0)
#CovY = np.cov(y,y,  ddof=0)
#
#R3 = CovXY/np.sqrt(CovX*CovY)
#
#plt.figure()
#plt.subplot(1,4,1)
#plt.imshow(R1)
#plt.subplot(1,4,2)
#plt.imshow(R2)
#plt.subplot(1,4,3)
#plt.imshow(R1-R2)
#plt.subplot(1,4,4)
#plt.imshow(R3)
#plt.show()
   
#a= obsData.Vx[:,0] 
#b= modData.Vx[:,0] 
#norm_a = np.linalg.norm(obsData.Vx[:,0])
#a = a / norm_a
#norm_b = np.linalg.norm(modData.Vx[:,0])
#b = a = b / norm_b
#c = np.correlate(a, b, mode = 'same')
#
#plt.figure()
#plt.plot(c)
#plt.show()



f = plt.figure(figsize=(20,10))
plt. subplot(2,3,1)
plt.plot(resData.Vx[:,0], label='resData.Vx')
plt.xticks([])
plt.title('station 1')
plt.legend(loc='lower right')

plt.ylabel('Amplitud', size=18)
plt. subplot(2,3,4)
plt.plot(resData.Vx[:,1], label='resData.Vx')
plt.xticks([])
plt.title('station 2')
plt.ylabel('Amplitud', size=18)
plt.legend(loc='lower right')

plt. subplot(2,3,2)
plt.plot(resData.Vy[:,0], label='resData.Vy')
plt.xticks([])
plt.title('station 1')
plt.legend(loc='lower right')

plt.ylabel('Amplitud', size=18)
plt. subplot(2,3,5)
plt.plot(resData.Vy[:,1], label='resData.Vy')
plt.xticks([])
plt.title('station 2')
plt.ylabel('Amplitud', size=18)
plt.legend(loc='lower right')

plt. subplot(2,3,3)
plt.plot(resData.Vz[:,0], label='resData.Vx')
plt.xticks([])
plt.title('station 1')
plt.legend(loc='lower right')

plt.ylabel('Amplitud', size=18)
plt. subplot(2,3,6)
plt.plot(resData.Vz[:,1], label='resData.Vy')
plt.xticks([])
plt.title('station 2')
plt.ylabel('Amplitud', size=18)
plt.legend(loc='lower right')


###############################################################################
###############################################################################
f = plt.figure(figsize=(20,10))
plt. subplot(1,3,1)
plt.title('resData.Vx')
plt.imshow(resData.Vx,  aspect='auto')
plt.xticks([])



plt. subplot(1,3,2)
plt.title('resData.Vy')
plt.imshow(resData.Vy, aspect='auto')
plt.xticks([])


plt. subplot(1,3,3)
plt.title('resData.Vz')
plt.imshow(resData.Vz, aspect='auto')
plt.xticks([])


###############################################################################
###############################################################################
f = plt.figure(figsize=(20,10))
plt. subplot(1,3,1)
plt.title('obsData.Vx')
plt.imshow(obsData.Vx,  aspect='auto')
plt.xticks([])



plt. subplot(1,3,2)
plt.title('obsData.Vy')
plt.imshow(obsData.Vy,  aspect='auto')
plt.xticks([])



plt. subplot(1,3,3)
plt.title('obsData.Vz')
plt.imshow(obsData.Vz,  aspect='auto')
plt.xticks([])




###############################################################################
f = plt.figure(figsize=(20,10))
plt. subplot(1,3,1)
plt.title('modData.Vx')
plt.imshow(modData.Vx, aspect='auto')
plt.xticks([])


plt. subplot(1,3,2)
plt.title('modData.Vy')
plt.imshow(modData.Vy, aspect='auto')
plt.xticks([])


plt. subplot(1,3,3)
plt.title('modData.Vz')
plt.imshow(modData.Vz,  aspect='auto')
plt.xticks([])

plt.show()

