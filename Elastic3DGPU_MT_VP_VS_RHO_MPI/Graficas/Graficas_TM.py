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
from math import sqrt, fabs
from numpy import linalg as LA
import pandas as pd
from scipy import signal
from obspy import *
from obspy.imaging.beachball import beach

""
class Gather():
    def __init__(self):
        self.Vx = None
        self.Vy = None
        self.Vz = None
            
class StressTensor():
    def __init__(self, Mxx,Myy, Mzz, Mxy, Mxz,  Myz):
        self.xx = Mxx
        self.yy = Myy
        self.zz = Mzz
        self.xy = Mxy
        self.xz = Mxz
        self.yz = Myz
    def __str__(self):
        return f"Mxx: {self.xx:.2f} Myy: {self.yy:.2f}  Mzz: {self.zz:.2f}  Mxy: {self.xy:.2f} Mxz: {self.xz:.2f} Myz: {self.yz:.2f}"


    def fromStressTensorToBeach(self, sta_x, sta_y,  facecolor, w, a):
        Mrr =  self.zz 
        Mtt =  self.xx 
        Mpp =  self.yy 
        Mrt =  self.xz 
        Mrp = -self.yz 
        Mtp = -self.xy 
        fm = [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]

        return beach(fm, xy=(sta_x, sta_y), facecolor=facecolor,  width=w, alpha=a)
    
    
dx = 1e3           # Paso espacial en x
dy = 1e3           # Paso espacial en y
dz = 1e3           # Paso espacial en z
fq = 0.1

plt.close ('all') 
path = 'Actual'
#path = 'Inversion_TM_Target_Mxx_1.0_Myy_1.0_Mzz_1.0_Mxy_0.0_Mxz_0.0_Myz_0.0_TM_2000ite'
path2 = 'Inversion_TM_Target_Mxx_1.0_Myy_0.0_Mzz_1.0_Mxy_0.0_Mxz_0.0_Myz_0.0_TM3D_2000ite_24Sismos'
pathTrue = 'Inversion_TM_Target_Mxx_1.0_Myy_1.0_Mzz_1.0_Mxy_0.0_Mxz_0.0_Myz_0.0_TrueModel'
sismo_pos_x = load(f'../Results/{path}/Geometry/sismos_pos_x.npy')
sismo_pos_y = load(f'../Results/{path}/Geometry/sismos_pos_y.npy')
sismo_pos_z = load(f'../Results/{path}/Geometry/sismos_pos_z.npy')
M = np.load(f'../Results/{path}/TM/targetTM.npy')
Mobs = StressTensor(M[0], M[1], M[2] , M[3], M[4], M[5])

''' Cargar Tensores '''
listTM = load(f'../Results/{path}/TM/listTM_0.1Hz.npy', allow_pickle=True)
listTM2 = load(f'../Results/{path}/TM/listTM_0.1Hz.npy', allow_pickle=True)
listTMtrue = load(f'../Results/{pathTrue}/TM/listTM_0.1Hz.npy', allow_pickle=True)
listTM2 = listTM2.copy()
nSismos = len(listTM)



f = plt.figure(figsize=(10, 5))
for src in range(nSismos):
    Mobs = StressTensor(M[0], M[1], M[2] , M[3], M[4], M[5])
    Mmod = StressTensor(listTM2[src].xx, listTM2[src].yy, listTM2[src].zz, listTM2[src].xy, listTM2[src].xz, listTM2[src].yz)
    Mini = StressTensor(0.5, 0.0, 0.5, 0.0, 0.5, 0.0)
    
    beachMobs = Mobs.fromStressTensorToBeach(src, 3.0, facecolor='r', w=1, a=1.0)
    beachMmod = Mmod.fromStressTensorToBeach(src, 1.5, facecolor='k', w=1, a=1.0)
    beachMini = Mini.fromStressTensorToBeach(src, 0, facecolor='b', w=1, a=1.0)

    ax = plt.gca()
    ax.set
    ax.add_collection(beachMobs) 
    ax.add_collection(beachMmod) 
    ax.add_collection(beachMini) 
    ax.set_aspect("equal")
    ax.set_ylim((-1, 4))
    ax.set_xlim((-1, nSismos))
#    ax.set_ylabel('$.$', size =14)
    ax.set_xlabel('$Earthquake$', size =14)
plt.show()
f.savefig("../pdf/Actual/BeachBall_FWITM_vs_TMtrue.pdf", bbox_inches='tight')



print('-'*10)
print('\nTensores Iniciales:')
for src in range(nSismos):
    print(f"Earthquake {src} -> {listTM[src]}")

print('\n===========>Errores  TM ================\n')    

fig, axes = plt.subplots(2,3, figsize=(20, 10), sharex = True, sharey = True)

for src in range(nSismos):
    
    axes[0][0].scatter(src, Mobs.xx, color='r', label=f'TM-true' if src==0 else None )
    #    axes[0][0].scatter(src, listTM[src].xx, color='b')
    axes[0][0].scatter(src, listTM2[src].xx, color='k', label=f'TM-FWI' if src==0 else None)
    axes[0][0].scatter(src, 0.5, color='b', label=f'TM-initial' if src==0 else None)
    #    axes[0][0].scatter(src, listTMtrue[src].xx, color='g')
    axes[0][0].set_ylabel('$Mxx$', size =14)
    axes[0][0].set_xlabel('$Earthquake$', size =14)
    axes[0][0].legend(loc='lower right')

    
    axes[0][1].scatter(src, Mobs.yy, color='r', label=f'TM-true' if src==0 else None )
#    axes[0][1].scatter(src, listTM[src].yy, color='b')
    axes[0][1].scatter(src, listTM2[src].yy, color='k', label=f'TM-FWI' if src==0 else None )
#    axes[0][1].scatter(src, listTMtrue[src].yy, color='g')
    axes[0][1].scatter(src, 0.5, color='b', label=f'TM-initial' if src==0 else None)
    axes[0][1].set_ylabel('$Myy$', size =14)
    axes[0][1].set_xlabel('$Earthquake$', size =14)
    axes[0][1].legend(loc='lower right')

    axes[0][2].scatter(src, Mobs.zz, color='r', label=f'TM-true' if src==0 else None )
#    axes[0][2].scatter(src, listTM[src].zz, color='b')
    axes[0][2].scatter(src, listTM2[src].zz, color='k', label=f'TM-FWI' if src==0 else None )
#    axes[0][2].scatter(src, listTMtrue[src].zz, color='g')
    axes[0][2].scatter(src, 0.5, color='b', label=f'TM-initial' if src==0 else None)
    axes[0][2].set_ylabel('$Mzz$', size =14)
    axes[0][2].set_xlabel('$Earthquake$', size =14)
    axes[0][2].legend(loc='lower right')
  
    axes[1][0].scatter(src, Mobs.xy, color='r', label=f'TM-true' if src==0 else None  )
#    axes[1][0].scatter(src, listTM[src].xy, color='b')
    axes[1][0].scatter(src, listTM2[src].xy, color='k', label=f'TM-FWI' if src==0 else None )
#    axes[1][0].scatter(src, listTMtrue[src].xy, color='g')
    axes[1][0].scatter(src, 0.5, color='b', label=f'TM-initial' if src==0 else None)
    axes[1][0].set_ylabel('$Mxy$', size =14)
    axes[1][0].set_xlabel('$Earthquake$', size =14)
    axes[1][0].legend(loc='upper right')

    axes[1][1].scatter(src, Mobs.xz, color='r', label=f'TM-true' if src==0 else None )
#    axes[1][1].scatter(src, listTM[src].xz, color='b')
    axes[1][1].scatter(src, listTM2[src].xz, color='k', label=f'TM-FWI' if src==0 else None )
#    axes[1][1].scatter(src, listTMtrue[src].xz, color='g')
    axes[1][1].scatter(src, 0.5, color='b', label=f'TM-initial' if src==0 else None)
    axes[1][1].set_ylabel('$Mxz$', size =14)
    axes[1][1].set_xlabel('$Earthquake$', size =14)
    axes[1][1].legend(loc='upper right')

    axes[1][2].scatter(src, Mobs.yz, color='r', label=f'TM-true' if src==0 else None )
#    axes[1][2].scatter(src, listTM[src].yz, color='b')
    axes[1][2].scatter(src, listTM2[src].yz, color='k', label=f'TM-FWI' if src==0 else None )
#    axes[1][2].scatter(src, listTMtrue[src].yz, color='g')
    axes[1][2].scatter(src, 0.5, color='b', label=f'TM-initial' if src==0 else None)
    axes[1][2].set_ylabel('$Myz$', size =14)
    axes[1][2].set_xlabel('$Earthquake$', size =14)
    axes[1][2].legend(loc='upper right')

fig.savefig("../pdf/Actual/FWITM_vs_TMtrue.pdf", bbox_inches='tight')

fig, axes = plt.subplots(2,3, figsize=(20, 10), sharex = True, sharey = True)

for src in range(nSismos):
    errorMxx = (fabs(Mobs.xx - listTM[src].xx))
    errorMyy = (fabs(Mobs.yy - listTM[src].yy))
    errorMzz = (fabs(Mobs.zz - listTM[src].zz))
    errorMxy = (fabs(Mobs.xy - listTM[src].xy))
    errorMxz = (fabs(Mobs.xz - listTM[src].xz))
    errorMyz = (fabs(Mobs.yz - listTM[src].yz))
    
    error2Mxx = (fabs(Mobs.xx - listTM2[src].xx))
    error2Myy = (fabs(Mobs.yy - listTM2[src].yy))
    error2Mzz = (fabs(Mobs.zz - listTM2[src].zz))
    error2Mxy = (fabs(Mobs.xy - listTM2[src].xy))
    error2Mxz = (fabs(Mobs.xz - listTM2[src].xz))
    error2Myz = (fabs(Mobs.yz - listTM2[src].yz))
    
    errorTrueMxx = (fabs(Mobs.xx - listTMtrue[src].xx))
    errorTrueMyy = (fabs(Mobs.yy - listTMtrue[src].yy))
    errorTrueMzz = (fabs(Mobs.zz - listTMtrue[src].zz))
    errorTrueMxy = (fabs(Mobs.xy - listTMtrue[src].xy))
    errorTrueMxz = (fabs(Mobs.xz - listTMtrue[src].xz))
    errorTrueMyz = (fabs(Mobs.yz - listTMtrue[src].yz))
    
#    axes[0][0].scatter(src, errorMxx, color='b')
    axes[0][0].scatter(src, error2Mxx, color='k', label=f'FWI' if src==0 else None)
    axes[0][0].scatter(src, 0.5, color='b', label=f'Initial' if src==0 else None)
#    axes[0][0].scatter(src, errorTrueMxx, color='g')
    axes[0][0].set_ylabel(r'$Absolute$ $error$ ${Mxx}$', size =14)
    axes[0][0].set_xlabel('$Earthquake$', size =14)
    axes[0][0].set_ylim([-0.1, 1])
    axes[0][0].legend(loc='upper right')

#    axes[0][1].scatter(src, errorMyy, color='b')
    axes[0][1].scatter(src, error2Myy, color='k', label=f'FWI' if src==0 else None)
    axes[0][1].scatter(src, 0.5, color='b', label=f'Initial' if src==0 else None)

#    axes[0][1].scatter(src, errorTrueMyy, color='g')
    axes[0][1].set_ylabel(r'$Absolute$ $error$ ${Myy}$', size =14)

    axes[0][1].set_xlabel('$Earthquake$', size =14)
    axes[0][1].set_ylim([-0.1, 1])
    axes[0][1].legend(loc='upper right')
#    axes[0][2].scatter(src, errorMzz, color='b')
    axes[0][2].scatter(src, error2Mzz, color='k', label=f'FWI' if src==0 else None)
    axes[0][2].scatter(src, 0.5, color='b', label=f'Initial' if src==0 else None)
#    axes[0][2].scatter(src, errorTrueMzz, color='g')
    axes[0][2].set_ylabel(r'$Absolute$ $error$ ${Mzz}$', size =14)
    axes[0][2].set_xlabel('$Earthquake$', size =14)
    axes[0][2].set_ylim([-0.1, 1])    
    axes[0][2].legend(loc='upper right')
#    axes[1][0].scatter(src, errorMxy, color='b')
    axes[1][0].scatter(src, error2Mxy, color='k', label=f'FWI' if src==0 else None)
    axes[1][0].scatter(src, 0.5, color='b', label=f'Initial' if src==0 else None)
#    axes[1][0].scatter(src, errorTrueMxy, color='g')
    axes[1][0].set_ylabel(r'$Absolute$ $error$ ${Mxy}$', size =14)
    axes[1][0].set_xlabel('$Earthquake$', size =14)
    axes[1][0].set_ylim([-0.1, 1])       
    axes[1][0].legend(loc='upper right')
#    axes[1][1].scatter(src, errorMxz, color='b')
    axes[1][1].scatter(src, error2Mxz, color='k', label=f'FWI' if src==0 else None)
    axes[1][1].scatter(src, 0.5, color='b', label=f'Initial' if src==0 else None)
#    axes[1][1].scatter(src, errorTrueMxz, color='g')
    axes[1][1].set_ylabel(r'$Absolute$ $error$ ${Mxz}$', size =14)
    axes[1][1].set_xlabel('$Earthquake$', size =14)
    axes[1][1].set_ylim([-0.1, 1])    
    axes[1][1].legend(loc='upper right')
#    axes[1][2].scatter(src, errorMyz, color='b')
    axes[1][2].scatter(src, error2Myz, color='k', label=f'FWI' if src==0 else None)
    axes[1][2].scatter(src, 0.5, color='b', label=f'Initial' if src==0 else None)
#    axes[1][2].scatter(src, errorTrueMyz, color='g')
    axes[1][2].set_ylabel(r'$Absolute$ $error$ ${Myz}$', size =14)
    axes[1][2].set_xlabel('$Earthquake$', size =14)
    axes[1][2].set_ylim([-0.1, 1])   
    axes[1][2].legend(loc='upper right')
    
#    
#    plt.subplot(2,3,3)
#    plt.scatter(src, Mobs.zz, color='r')
#    plt.scatter(src, listTM[src].zz, color='b')
#    plt.title('Mzz')
    print(f"Error Earthquake {src} -> Mxx: {error2Mxx:.2f} Myy: {error2Myy:.2f} Mzz: {error2Mzz:.2f} Mxy: {error2Mxy:.2f} Mxz: {error2Mxz:.2f} Myz: {error2Myz:.2f}")    
fig.savefig("../pdf/Actual/AbsoluteError_TM.pdf", bbox_inches='tight')
        
#listTM = []
#for idxSismo in range(nSismos):
#    Mmodxx = load(f'../Results/{path}/TM/Mmodxx{idxSismo}.npy')[-1]
#    Mmodyy = load(f'../Results/{path}/TM/Mmodyy{idxSismo}.npy')[-1]
#    Mmodzz = load(f'../Results/{path}/TM/Mmodzz{idxSismo}.npy')[-1]
#    Mmodxy = load(f'../Results/{path}/TM/Mmodxy{idxSismo}.npy')[-1]
#    Mmodxz = load(f'../Results/{path}/TM/Mmodxz{idxSismo}.npy')[-1]
#    Mmodyz = load(f'../Results/{path}/TM/Mmodyz{idxSismo}.npy')[-1]
#    M = StressTensor(Mmodxx, Mmodyy, Mmodzz , Mmodxy, Mmodxz, Mmodyz)
#    listTM.append(M)
#save(f'../Results/{path}/TM/listTM_{fq}Hz.npy', listTM)

idxSismo = 2
costFunction = load(f'../Results/{path}/CostFunction/costFunction{idxSismo}.npy')
Mmodxx = load(f'../Results/{path}/TM/Mmodxx{idxSismo}.npy')
Mmodyy = load(f'../Results/{path}/TM/Mmodyy{idxSismo}.npy')
Mmodzz = load(f'../Results/{path}/TM/Mmodzz{idxSismo}.npy')
Mmodxy = load(f'../Results/{path}/TM/Mmodxy{idxSismo}.npy')
Mmodxz = load(f'../Results/{path}/TM/Mmodxz{idxSismo}.npy')
Mmodyz = load(f'../Results/{path}/TM/Mmodyz{idxSismo}.npy')
VpTarget = load(f'../Results/{path}/Models/VpTarget.npy')

ny, nz, nx = VpTarget.shape

f = plt.figure(figsize=(20,10))
plt.subplot(2,3,1),
plt.plot(Mmodxx, label= f"FWI:{Mmodxx[-1]:.3f}")
plt.xlabel('Iteraciones', size=18)
plt.axhline(y=Mobs.xx, xmin=0.1, xmax=len(Mmodxx)-2, color='r', label= f"Target:{Mobs.xx}")
plt.ylabel('Mmodxx', size=18)
plt.legend()

plt.subplot(2,3,2),
plt.plot(Mmodyy, label= f"FWI:{Mmodyy[-1]:.3f}")
plt.xlabel('Iteraciones', size=18)
plt.axhline(y=Mobs.yy, xmin=0.1, xmax=len(Mmodzz)-2, color='r', label= f"Target:{Mobs.yy}")
plt.ylabel('Mmodyy', size=18)
plt.legend()

plt.subplot(2,3,3),
plt.plot(Mmodzz, label= f"FWI:{Mmodzz[-1]:.3f}")
plt.xlabel('Iteraciones', size=18)
plt.axhline(y=Mobs.zz, xmin=0.1, xmax=len(Mmodxz)-2, color='r', label= f"Target:{Mobs.zz}")
plt.ylabel('Mmodzz', size=18)
plt.legend()

plt.subplot(2,3,4),
plt.plot(Mmodxy, label= f"FWI:{Mmodxy[-1]:.3f}")
plt.xlabel('Iteraciones', size=18)
plt.axhline(y=Mobs.xy, xmin=0.1, xmax=len(Mmodxx)-2, color='r', label= f"Target:{Mobs.xy}")
plt.ylabel('Mmodxy', size=18)
plt.legend()

plt.subplot(2,3,5),
plt.plot(Mmodxz, label= f"FWI:{Mmodxz[-1]:.3f}")
plt.xlabel('Iteraciones', size=18)
plt.axhline(y=Mobs.xz, xmin=0.1, xmax=len(Mmodzz)-2, color='r', label= f"Target:{Mobs.xz}")
plt.ylabel('Mmodxz', size=18)
plt.legend()

plt.subplot(2,3,6),
plt.plot(Mmodyz, label= f"FWI:{Mmodyz[-1]:.3f}")
plt.xlabel('Iteraciones', size=18)
plt.axhline(y=Mobs.yz, xmin=0.1, xmax=len(Mmodxz)-2, color='r', label= f"Target:{Mobs.yz}")
plt.ylabel('Mmodyz', size=18)
plt.legend()
f.savefig("../pdf/Actual/FWITM3D.pdf", bbox_inches='tight')

f = plt.figure()
plt.plot(costFunction)
plt.xlabel('Iteraciones', size=18)
plt.ylabel('Funcion de Costo', size=18)
f.savefig("../pdf/Actual/CostFunction.pdf", bbox_inches='tight')



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


modData0.Vx = load(f'../Results/{path}/Gathers/modDataIni{idxSismo}.Vx.npy')
modData0.Vy = load(f'../Results/{path}/Gathers/modDataIni{idxSismo}.Vy.npy')
modData0.Vz= load(f'../Results/{path}/Gathers/modDataIni{idxSismo}.Vz.npy')

modData.Vx = load(f'../Results/{path}/Gathers/modData{idxSismo}.Vx.npy')
modData.Vy = load(f'../Results/{path}/Gathers/modData{idxSismo}.Vy.npy')
modData.Vz= load(f'../Results/{path}/Gathers/modData{idxSismo}.Vz.npy')

obsData.Vx = load(f'../Results/{path}/Gathers/obsData{idxSismo}.Vx.npy')
obsData.Vy = load(f'../Results/{path}/Gathers/obsData{idxSismo}.Vy.npy')
obsData.Vz= load(f'../Results/{path}/Gathers/obsData{idxSismo}.Vz.npy')

resData.Vx = load(f'../Results/{path}/Gathers/resData{idxSismo}.Vx.npy')
resData.Vy = load(f'../Results/{path}/Gathers/resData{idxSismo}.Vy.npy')
resData.Vz= load(f'../Results/{path}/Gathers/resData{idxSismo}.Vz.npy')

###############################################################################
f = plt.figure(figsize=(20,10))
plt. subplot(2,3,1)
plt.plot(obsData.Vx[:,0], label='obsData.Vx')
plt.plot(modData.Vx[:,0], label='modData.Vx FWI')
plt.plot(modData0.Vx[:,0], label='modData.Vx ite 1')
plt.xticks([])
plt.title('station 1')
plt.legend(loc='lower right')

plt.ylabel('Amplitud', size=18)
plt. subplot(2,3,4)
plt.plot(obsData.Vx[:,1], label='obsData.Vx')
plt.plot(modData.Vx[:,1], label='modData.Vx FWI')
plt.plot(modData0.Vx[:,1], label='modData.Vx ite 1')
plt.xticks([])
plt.title('station 2')
plt.ylabel('Amplitud', size=18)
plt.legend(loc='lower right')


###############################################################################


plt. subplot(2,3,2)
plt.plot(obsData.Vy[:,0], label='obsData.Vy')
plt.plot(modData.Vy[:,0], label='modData.Vy FWI')
plt.plot(modData0.Vy[:,0], label='modData.Vy ite 1')
plt.xticks([])
plt.title('station 1')
plt.legend(loc='lower right')

plt. subplot(2,3,5)
plt.plot(obsData.Vy[:,1], label='obsData.Vy')
plt.plot(modData.Vy[:,1], label='modData.Vy FWI')
plt.plot(modData0.Vy[:,1], label='modData.Vy ite 1')
plt.xticks([])
plt.title('station 2')
plt.legend(loc='lower right')


###############################################################################
plt. subplot(2,3,3)
plt.plot(obsData.Vz[:,0], label='obsData.Vz')
plt.plot(modData.Vz[:,0], label='modData.Vz FWI')
plt.plot(modData0.Vz[:,0], label='modData.Vz ite 1')
plt.title('station 1')
plt.xticks([])
plt.legend(loc='lower right')

plt. subplot(2,3,6)
plt.plot(obsData.Vz[:,1], label='obsData.Vz')
plt.plot(modData.Vz[:,1], label='modData.Vz FWI')
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

