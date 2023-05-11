#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:08:41 2022

@author: jheyston
"""
import pycuda.autoinit
from scipy.ndimage import gaussian_filter
from numpy import load, save
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from tqdm import tqdm
from math import sqrt

""


class Gather():
    def __init__(self):
        self.Vx = None
        self.Vz = None


class MomentoTensor():
    def __init__(self, Mxx, Mzz, Mxz):
        self.xx = Mxx
        self.zz = Mzz
        self.xz = Mxz

    def __str__(self):
        return f"Mxx: {self.xx:.2f} Mzz: {self.zz:.2f} Mxz: {self.xz:.2f}"


def graficarModel(model1, name1, model2, name2, model3, name3, fq, vmin, vmax):
    im_ratio = model1.shape[0]/model1.shape[1]
    ny, nz, nx = model1.shape
    paso = 20
    xticks = np.arange(0, (nx-2*lenPML), paso)
    yticks = np.arange(0, (ny-2*lenPML), paso)
    zticks = np.arange(0, (nz-lenPML), paso)
    
    corte_x = corteY
    corte_y = 52
    corte_z = corteY
    
    xlabel_list = np.arange(lenPML, (nx-lenPML)*(dx/1e3), paso*(dx/1e3))
    ylabel_list = np.arange(lenPML, (ny-lenPML)*(dy/1e3), paso*(dy/1e3))
    zlabel_list = np.arange(0, (nz-lenPML)*(dz/1e3), paso*(dz/1e3))


    
    fig, axes = plt.subplots(3,3, figsize=(10, 10))
    

    axes[0][0].imshow(model1[lenPML+2:ny-lenPML-2,:nz-lenPML-2,corte_x].T, vmin=vmin, vmax=vmax)
    axes[0][0].set_xticks(list(yticks))
    axes[0][0].set_xticklabels(list(ylabel_list.astype(int)))
    axes[0][0].set_yticks(list(zticks))
    axes[0][0].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[0][0].set_ylabel('$Depth (km)$', size =14)
#    axes[0][0].set_xlabel('$Distance (Km)$', size =14)
    axes[0][0].set_title('$a)$', size=14)
    
    axes[0][1].imshow(model1[corte_y, 0:nz-lenPML-2,lenPML+3:nx-lenPML-2], vmin=vmin, vmax=vmax)
    axes[0][1].set_xticks(list(xticks))
    axes[0][1].set_xticklabels(list(xlabel_list.astype(int)))
    axes[0][1].set_yticks(list(zticks))
    axes[0][1].set_yticklabels(list(zlabel_list.astype(int)))
   
#    axes[0][1].set_ylabel('$Depth (Km)$', size =14)
#    axes[0][1].set_xlabel('$Distance (Km)$', size =14)
    axes[0][1].set_title('$b)$', size=14)
    
    axes[0][2].imshow(model1[lenPML+3:ny-lenPML-2, corte_z, lenPML+3:nx-lenPML-2], vmin=vmin, vmax=vmax)
    axes[0][2].set_xticks(list(xticks))
    axes[0][2].set_xticklabels(list(xlabel_list.astype(int)))
    axes[0][2].set_yticks(list(yticks))
    axes[0][2].set_yticklabels(list(ylabel_list.astype(int)))
   
#    axes[0][2].set_ylabel('$Depth (Km)$', size =14)
#    axes[0][2].set_xlabel('$Distance (Km)$', size =14)
    axes[0][2].set_title('$c)$', size=14)
    
    axes[1][0].imshow(model2[lenPML+2:ny-lenPML-2,:nz-lenPML-2,corte_x].T, vmin=vmin, vmax=vmax)
    axes[1][0].set_xticks(list(yticks))
    axes[1][0].set_xticklabels(list(ylabel_list.astype(int)))
    axes[1][0].set_yticks(list(zticks))
    axes[1][0].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[1][0].set_ylabel('$Depth (km)$', size =14)
#    axes[1][0].set_xlabel('$Distance (Km)$', size =14)
    axes[1][0].set_title('$d)$', size=14)
    
    axes[1][1].imshow(model2[corte_y, 0:nz-lenPML-2,lenPML+3:nx-lenPML-2], vmin=vmin, vmax=vmax)
    axes[1][1].set_xticks(list(xticks))
    axes[1][1].set_xticklabels(list(xlabel_list.astype(int)))
    axes[1][1].set_yticks(list(zticks))
    axes[1][1].set_yticklabels(list(zlabel_list.astype(int)))
   
#    axes[1][1].set_ylabel('$Depth (Km)$', size =14)
#    axes[1][1].set_xlabel('$Distance (Km)$', size =14)
    axes[1][1].set_title('$e)$', size=14)
    
    axes[1][2].imshow(model2[lenPML+3:ny-lenPML-2, corte_z, lenPML+3:nx-lenPML-2], vmin=vmin, vmax=vmax)
    axes[1][2].set_xticks(list(xticks))
    axes[1][2].set_xticklabels(list(xlabel_list.astype(int)))
    axes[1][2].set_yticks(list(yticks))
    axes[1][2].set_yticklabels(list(ylabel_list.astype(int)))
   
#    axes[1][2].set_ylabel('$Depth (Km)$', size =14)
#    axes[1][2].set_xlabel('$Distance (Km)$', size =14)
    axes[1][2].set_title('$f)$', size=14)
    
    axes[2][0].imshow(model3[lenPML+2:ny-lenPML-2,:nz-lenPML-2,corte_x].T, vmin=vmin, vmax=vmax)
    axes[2][0].set_xticks(list(yticks))
    axes[2][0].set_xticklabels(list(ylabel_list.astype(int)))
    axes[2][0].set_yticks(list(zticks))
    axes[2][0].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[2][0].set_ylabel('$Depth (km)$', size =14)
    axes[2][0].set_xlabel('$Distance (km)$', size =14)
    axes[2][0].set_title('$g)$', size=14)
    
    axes[2][1].imshow(model3[corte_y, 0:nz-lenPML-2,lenPML+3:nx-lenPML-2], vmin=vmin, vmax=vmax)
    axes[2][1].set_xticks(list(xticks))
    axes[2][1].set_xticklabels(list(xlabel_list.astype(int)))
    axes[2][1].set_yticks(list(zticks))
    axes[2][1].set_yticklabels(list(zlabel_list.astype(int)))
   
#    axes[2][1].set_ylabel('$Depth (Km)$', size =14)
    axes[2][1].set_xlabel('$Distance (km)$', size =14)
    axes[2][1].set_title('$h)$', size=14)
    
    axes[2][2].imshow(model3[lenPML+3:ny-lenPML-2, corte_z, lenPML+3:nx-lenPML-2], vmin=vmin, vmax=vmax)
    axes[2][2].set_xticks(list(xticks))
    axes[2][2].set_xticklabels(list(xlabel_list.astype(int)))
    axes[2][2].set_yticks(list(yticks))
    axes[2][2].set_yticklabels(list(ylabel_list.astype(int)))
   
#    axes[2][2].set_ylabel('$Depth (Km)$', size =14)
    axes[2][2].set_xlabel('$Distance (km)$', size =14)
    axes[2][2].set_title('$i)$', size=14)
       
#    im = axes[1].imshow(model2, vmin=vmin, vmax=vmax)
#    axes[1].set_xticks(list(xticks))
#    axes[1].set_xticklabels(list(xlabel_list.astype(int)))
#    axes[1].set_yticks(list(zticks))
#    axes[1].set_yticklabels(list(zlabel_list.astype(int)))
#    
#    axes[1].set_ylabel('$Depth (Km)$', size =14)
#    axes[1].set_xlabel('$Distance (Km)$', size =14)
#    axes[1].set_title('$b)$', size=14)
#    
#    
#    im = axes[2].imshow(model3, vmin=vmin, vmax=vmax)
#    
#    axes[2].set_xticks(list(xticks))
#    axes[2].set_xticklabels(list(xlabel_list.astype(int)))
#    axes[2].set_yticks(list(zticks))
#    axes[2].set_yticklabels(list(zlabel_list.astype(int)))
#    
#    axes[2].set_ylabel('$Depth (Km)$', size =14)
#    axes[2].set_xlabel('$Distance (Km)$', size =14)
#    axes[2].set_title('$c)$', size=14)
#    plt.colorbar(im, fraction=0.047*im_ratio)
    
    plt.show()
    fig.savefig(f"../pdf/Actual/{name1}_and_{name2}_{fq}Hz.pdf", bbox_inches='tight')



if __name__ == "__main__":

    plt.close ('all') 

    # ---------------------------------------------------------------------------------------------------------------- #
    ''' Parametros de inicio '''
    fq = 0.1
    dx = 1e3
    dy = dx
    dz = dx
    sismo = 5
    corteY = 50
    corteX = [40, 50, 60]
    lenPML = 20
    # -----------------------------------------------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#

    path = "Actual"
#    path = 'FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_2000ite'
#    path = 'FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite_GeomSGC'
#    path = 'FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite_GeomTemporal'
#    path = 'FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite_Geom_SGC_Temporal_Permanente'
    
    
    ''' Carga de archivos '''
    costFunction = load(f'../Results/{path}/CostFunction/costFunction_{fq}Hz.npy')
    VpTarget = load(f'../Results/{path}/Models/VpTarget.npy')
    VsTarget = load(f'../Results/{path}/Models/VsTarget.npy')
    RhoTarget = load(f'../Results/{path}/Models/RhoTarget.npy')
    LambdaTarget = load(f'../Results/{path}/Models/LambdaTarget.npy')
    MuTarget = load(f'../Results/{path}/Models/MuTarget.npy')
    VpInicial = load(f'../Results/{path}/Models/VpInicial.npy')
    VsInicial = load(f'../Results/{path}/Models/VsInicial.npy')
    RhoInicial = load(f'../Results/{path}/Models/RhoInicial.npy')
    LambdaInicial = load(f'../Results/{path}/Models/LambdaInicial.npy')
    MuInicial = load(f'../Results/{path}/Models/MuInicial.npy')
    Vp = load(f'../Results/{path}/Models/Vp_{fq}Hz.npy')
    Vs = load(f'../Results/{path}/Models/Vs_{fq}Hz.npy')
    Rho = load(f'../Results/{path}/Models/Rho_{fq}Hz.npy')
    Lambda = load(f'../Results/{path}/Models/Lambda_{fq}Hz.npy')
    Mu = load(f'../Results/{path}/Models/Mu_{fq}Hz.npy')
    modData = np.load(f'../Results/{path}/Gathers/modData_{fq}Hz.npy', allow_pickle=True)
    obsData = np.load(f'../Results/{path}/Gathers/obsData_{fq}Hz.npy', allow_pickle=True)
    sismo_pos_x = load(f'../Results/{path}/Geometry/sismos_pos_x.npy')
    sismo_pos_y = load(f'../Results/{path}/Geometry/sismos_pos_y.npy')
    sismo_pos_z = load(f'../Results/{path}/Geometry/sismos_pos_z.npy')
    sta_pos_x = load(f'../Results/{path}/Geometry/sta_pos_x.npy')
    sta_pos_y = load(f'../Results/{path}/Geometry/sta_pos_y.npy')
    sta_pos_z = load(f'../Results/{path}/Geometry/sta_pos_z.npy')

    # -----------------------------------------------------------------------------------------------------------------#
    ny, nz, nx = VpTarget.shape
#    plt.figure()
#    plt.imshow(obsData[1].Vz, aspect='auto')
#    plt.show()
    
    paso = 10
    xticks = np.arange(0, (nx), paso)
    zticks = np.arange(0, -(nz), -paso)

    xlabel_list = np.arange(0, (nx)*(dx/1e3), paso*(dx/1e3))
    zlabel_list = np.arange(0, (nz)*(dz/1e3), paso*(dz/1e3))


    # Set up a figure twice as tall as it is wide
    fig = plt.figure(figsize=(12,5))
#    fig.suptitle('A tale of 2 subplots')
    
    # First subplot
    ax = fig.add_subplot(1, 2, 2)
    
    ax.scatter(sta_pos_x[:20], -sta_pos_z[:20], marker ='^', edgecolors='k', s=30, color="r", label='Stations')
    ax.scatter(sismo_pos_x[-8:], -sismo_pos_z[-8:], edgecolors='k', s=30, color="b", label='Earthquakes')
    ax.set_xlabel('$Distance (km)$', size=18)
    ax.set_ylabel('$Depth (km)$', size=18)
    ax.set_xticks(list(xticks))
    ax.set_xticklabels(list(xlabel_list.astype(int)))
    ax.set_yticks(list(zticks))
    ax.set_yticklabels(list(zlabel_list.astype(int)))
    ax.set_title('$b)$', size=18)
    plt.legend()
    
    # Second subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    
    x = np.linspace(0, 100, 10)
    z = np.arange(0, -100, -10)
    X, Z = np.meshgrid(x, z)
    Y = 52
    
    ax.scatter3D(sta_pos_x, sta_pos_y, -sta_pos_z, marker ='^', edgecolors='k', s=30, color="r", label='Stations')
    ax.scatter3D(sismo_pos_x, sismo_pos_y, -sismo_pos_z, edgecolors='k', s=30, color="b", label='Earthquakes')
    surf = ax.plot_surface(X, Y, Z, alpha=0.2, color='g')
    ax.set_xlabel(r'$X (km)$', size=18)
    ax.set_ylabel(r'$Y (km)$', size=18)
    ax.set_zlabel(r'$Z (km)$', size=18)
    ax.set_title('$a)$', size=18)
    ax.legend()
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(-nz, 0))
    
    plt.show()

    fig.savefig(f"../pdf/Actual/Geometry2D_3D_seismology.pdf", bbox_inches='tight')
    # -----------------------------------------------------------------------------------------------------------------#

    nSismos = len(obsData)
    nt, nStations = modData[sismo].Vx.shape

    print(nt, nStations)
    
    vminVp = np.amin(VpTarget)
    vmaxVp = np.amax(VpTarget)

    vminVs = np.amin(VsTarget)
    vmaxVs = np.amax(VsTarget)

    vminRho = np.amin(RhoTarget)
    vmaxRho = np.amax(RhoTarget)

    vminLambda = np.amin(LambdaTarget)
    vmaxLambda = np.amax(LambdaTarget)

    vminMu = np.amin(MuTarget)
    vmaxMu = np.amax(MuTarget)
    for iy in range(ny):
        for iz in range(nz):
            for ix in range(nx):
                if Mu[iy, iz, ix]< 0:
 
                    Mu[iy,iz, ix] = MuInicial[iy,iz, ix] 
                        
    Vs = np.sqrt((Mu)/Rho)
    Vp = np.sqrt((Lambda+2*Mu)/Rho)
    
#    Mu = Vs.copy() * Vs.copy() * Rho.copy()
#    Lambda = Vp.copy() * Vp.copy() * Rho.copy() - 2.0 * Mu.copy()
#    Rho = (1.6612 * ( Vp.copy() / 1e3) - 0.4712 * ((Vp.copy() / 1e3) ** 2) + 0.0671 * ((Vp.copy() / 1e3) ** 3) - 0.0043 * (
#                 (Vp.copy() / 1e3) ** 4) + 0.000106 * ((Vp.copy() / 1e3) ** 5))*1e3
#    Rho[:, :10, :] = RhoTarget[:, :10, :]

    graficarModel(VsTarget, 'VsTarget', VsInicial, 'VsInicial', Vs, 'Vs',   fq, vminVs, vmaxVs)
    graficarModel(VpTarget, 'VpTarget',  VpInicial, 'VpInicial', Vp, 'Vp',  fq, vminVp, vmaxVp)
    graficarModel(RhoTarget, 'RhoTarget', RhoInicial, 'RhoInicial', Rho, 'Rho',  fq, vminRho, vmaxRho)
    graficarModel(LambdaTarget, 'LambdaTarget', LambdaInicial, 'LambdaInicial', Lambda, 'Lambda',  fq, vminLambda, vmaxLambda)
    graficarModel(MuTarget, 'MuTarget', MuInicial,  'MuInicial', Mu, 'Mu', fq, vminMu, vmaxMu)
    
    


#    f = plt.figure()
    fig, ax = plt.subplots(3, 3, sharex = True, figsize=(20,10))
    zticks = np.arange(0, (nz), paso)

#    plt.subplot(3,2,2)
    ax[0][0].plot(VpTarget[corteY, :,  corteX[0]], label='$Real~ Vp$')
    ax[0][0].plot(Vp[corteY, :,  corteX[0]], label='$FWI~ Vp$')
    ax[0][0].plot(VpInicial[corteY, :,  corteX[0]], label='$Initial~ Vp$')
    ax[0][0].set_ylabel('$Vp(m/s)$', size=14)
    ax[0][0].set_xlabel('$Depth (km)$', size=14)
    ax[0][0].set_title('$a)$', loc='left', size=18)    
    ax[0][0].legend(loc='lower left')
    ax[0][0].set_xticks(list(zticks))
    ax[0][0].set_xticklabels(list(zlabel_list.astype(int)))
    ax[0][0].set_ylim([1200, 5000])
    
    ax[1][0].plot(VpTarget[corteY, :,  corteX[1]], label='$Real~ Vs$')
    ax[1][0].plot(Vp[corteY, :, corteX[1]], label='$FWI~ Vs$')
    ax[1][0].plot(VpInicial[corteY, :,  corteX[1]], label='$Initial~ Vs$')
    ax[1][0].set_ylabel('$Vp(m/s)$', size=14)
    ax[1][0].set_xlabel('$Depth (km)$', size=14)
    ax[1][0].set_title('$d)$', loc='left', size=18)  
    ax[1][0].set_xticks(list(zticks))
    ax[1][0].set_xticklabels(list(zlabel_list.astype(int)))
    ax[1][0].legend(loc='lower left')
    ax[1][0].set_ylim([1200, 5000])

    ax[2][0].plot(VpTarget[corteY, :,  corteX[2]], label='$Real~ Vp$')
    ax[2][0].plot(Vp[corteY, :,  corteX[2]], label='$FWI~ Vp$')
    ax[2][0].plot(VpInicial[corteY, :, corteX[2]], label='$Initial~ Vp$')
    ax[2][0].set_ylabel('$Vp(m/s)$', size=14)
    ax[2][0].set_xlabel('$Depth (km)$', size=14)
    ax[2][0].set_title('$g)$', loc='left', size=18)    
    ax[2][0].set_xticks(list(zticks))
    ax[2][0].set_xticklabels(list(zlabel_list.astype(int)))
    ax[2][0].legend(loc='lower left')
    ax[2][0].set_ylim([1200, 5000])

    
    ax[0][1].plot(VsTarget[corteY, :,  corteX[0]], label='$Real~ Vs$')
    ax[0][1].plot(Vs[corteY, :,  corteX[0]], label='$FWI~ Vs$')
    ax[0][1].plot(VsInicial[corteY, :,  corteX[0]], label='$Initial~ Vs$')
    ax[0][1].set_ylabel('$Vs(m/s)$', size=14)
    ax[0][1].set_xlabel('$Depth (km)$', size=14)
    ax[0][1].set_title('$b)$', loc='left', size=18)    
    ax[0][1].legend(loc='lower left')
    ax[0][1].set_ylim([600, 2600])
    
    ax[1][1].plot(VsTarget[corteY, :,  corteX[1]], label='$Real~ Vs$')
    ax[1][1].plot(Vs[corteY, :,  corteX[1]], label='$FWI~ Vs$')
    ax[1][1].plot(VsInicial[corteY, :,  corteX[1]], label='$Initial~ Vs$')
    ax[1][1].set_ylabel('$Vs(m/s)$', size=14)
    ax[1][1].set_xlabel('$Depth (km)$', size=14)
    ax[1][1].set_title('$e)$', loc='left', size=18)    
    ax[1][1].legend(loc='lower left')
    ax[1][1].set_ylim([600, 2600])
    
    ax[2][1].plot(VsTarget[corteY, :,  corteX[2]], label='$Real~ Vs$')
    ax[2][1].plot(Vs[corteY, :,  corteX[2]], label='$FWI~ Vs$')
    ax[2][1].plot(VsInicial[ corteY, :,  corteX[2]], label='$Initial~ Vs$')
    ax[2][1].set_ylabel('$Vs(m/s)$', size=14)
    ax[2][1].set_xlabel('$Depth (km)$', size=14)
    ax[2][1].set_title('$h)$', loc='left', size=18)    
    ax[2][1].set_xticks(list(zticks))
    ax[2][1].set_xticklabels(list(zlabel_list.astype(int)))
    ax[2][1].legend(loc='lower left')
    ax[2][1].set_ylim([600, 2600])

    ax[0][2].plot(RhoTarget[corteY, :, corteX[0]], label=r'$Real~ \rho$')
    ax[0][2].plot(Rho[corteY, :,  corteX[0]], label=r'$FWI~ \rho$')
    ax[0][2].plot(RhoInicial[corteY, :,  corteX[0]], label=r'$Initial~ \rho$')
    ax[0][2].set_ylabel(r'$\rho (kg/m^3)$', size=14)
    ax[0][2].set_xlabel('$Depth (km)$', size=14)
    ax[0][2].set_title('$c)$', loc='left', size=18)    
    ax[0][2].legend(loc='lower left')
    ax[0][2].set_ylim([1300, 3000])

    
    ax[1][2].plot(RhoTarget[corteY,  :, corteX[1]], label=r'$Real~ \rho$')
    ax[1][2].plot(Rho[ corteY, :, corteX[1]], label=r'$FWI~ \rho$')
    ax[1][2].plot(RhoInicial[corteY, :, corteX[1]], label=r'$Initial~ \rho$')
    ax[1][2].set_ylabel(r'$\rho (kg/m^3)$', size=14)
    ax[1][2].set_xlabel('$Depth (km)$', size=14)
    ax[1][2].set_title('$f)$', loc='left', size=18)    
    ax[1][2].legend(loc='lower left')
    ax[1][2].set_ylim([1300, 3000])


    ax[2][2].plot(RhoTarget[corteY, :, corteX[2]], label=r'$Real~ \rho$')
    ax[2][2].plot(Rho[corteY, :,  corteX[2]], label=r'$FWI~ \rho$')
    ax[2][2].plot(RhoInicial[corteY, :,  corteX[2]], label=r'$Initial~ \rho$')
    ax[2][2].set_ylabel(r'$\rho (kg/m^3)$', size=14)
    ax[2][2].set_xlabel('$Depth (km)$', size=14)
    ax[2][2].set_title('$i)$', loc='left', size=18)   
    ax[2][2].set_xticks(list(zticks))
    ax[2][2].set_xticklabels(list(zlabel_list.astype(int)))
    ax[2][2].legend(loc='lower left')
    ax[2][2].set_ylim([1300, 3000])
   
    fig.savefig(f"../pdf/Actual/corteVp_Vs_{fq}Hz.pdf", bbox_inches='tight')

    

    f = plt.figure()
    plt.plot(costFunction)
    plt.ylabel('$Cost~function$', size=14)
    plt.xlabel('$Iterations$', size=14)
    f.savefig(f"../pdf/Actual/costFunction_{fq}Hz.pdf", bbox_inches='tight')


    fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize=(10,10))

    ax[0][0].plot(obsData[sismo].Vx[:, (nStations // 5)], label='obs')
    ax[0][0].plot(modData[sismo].Vx[:, (nStations // 5)], label='mod')
    ax[0][0].set_xlabel('$Samples$', size=14)
    ax[0][0].set_ylabel('$Vx~amplitude$', size=14)
    ax[0][0].set_title('$a)$', loc='center', size=18)    
    ax[0][0].legend()
    
    ax[0][1].plot(obsData[sismo].Vx[:, 2*(nStations // 5)], label='obs')
    ax[0][1].plot(modData[sismo].Vx[:, 2*(nStations // 5)], label='mod')
    ax[0][1].set_xlabel('$Samples$', size=14)
    ax[0][1].set_ylabel('$Vx~amplitude$', size=14)
    ax[0][1].set_title('$b)$', loc='center', size=18)    
    ax[0][1].legend()
    
    ax[1][0].plot(obsData[sismo].Vx[:, 3*(nStations // 5)], label='obs')
    ax[1][0].plot(modData[sismo].Vx[:, 3*(nStations // 5)], label='mod')
    ax[1][0].set_xlabel('$Samples$', size=14)
    ax[1][0].set_ylabel('$Vx~amplitude$', size=14)
    ax[1][0].set_title('$c)$', loc='center', size=18)    
    ax[1][0].legend()
    
    ax[1][1].plot(obsData[sismo].Vx[:, 4*(nStations // 5)], label='obs')
    ax[1][1].plot(modData[sismo].Vx[:, 4*(nStations // 5)], label='mod')
    ax[1][1].set_xlabel('$Samples$', size=14)
    ax[1][1].set_ylabel('$Vx~amplitude$', size=14)
    ax[1][1].set_title('$d)$', loc='center', size=18)    
    ax[1][1].legend()
    
    
    fig.savefig(f"../pdf/Actual/Comparacion_obsData_modData_Vx_{fq}Hz.pdf", bbox_inches='tight')
    
    
    fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize=(10,10))

    ax[0][0].plot(obsData[sismo].Vy[:, (nStations // 5)], label='obs')
    ax[0][0].plot(modData[sismo].Vy[:, (nStations // 5)], label='mod')
    ax[0][0].set_xlabel('$Samples$', size=14)
    ax[0][0].set_ylabel('$Vy~amplitude$', size=14)
    ax[0][0].set_title('$a)$', loc='center', size=18)    
    ax[0][0].legend()
    
    ax[0][1].plot(obsData[sismo].Vy[:, 2*(nStations // 5)], label='obs')
    ax[0][1].plot(modData[sismo].Vy[:, 2*(nStations // 5)], label='mod')
    ax[0][1].set_xlabel('$Samples$', size=14)
    ax[0][1].set_ylabel('$Vy~amplitude$', size=14)
    ax[0][1].set_title('$b)$', loc='center', size=18)    
    ax[0][1].legend()
    
    ax[1][0].plot(obsData[sismo].Vy[:, 3*(nStations // 5)], label='obs')
    ax[1][0].plot(modData[sismo].Vy[:, 3*(nStations // 5)], label='mod')
    ax[1][0].set_xlabel('$Samples$', size=14)
    ax[1][0].set_ylabel('$Vy~amplitude$', size=14)
    ax[1][0].set_title('$c)$', loc='center', size=18)    
    ax[1][0].legend()
    
    ax[1][1].plot(obsData[sismo].Vy[:, 4*(nStations // 5)], label='obs')
    ax[1][1].plot(modData[sismo].Vy[:, 4*(nStations // 5)], label='mod')
    ax[1][1].set_xlabel('$Samples$', size=14)
    ax[1][1].set_ylabel('$Vy~amplitude$', size=14)
    ax[1][1].set_title('$d)$', loc='center', size=18)    
    ax[1][1].legend()
    
    
    fig.savefig(f"../pdf/Actual/Comparacion_obsData_modData_Vy_{fq}Hz.pdf", bbox_inches='tight')
    
    fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize=(10,10))

    ax[0][0].plot(obsData[sismo].Vz[:, (nStations // 5)], label='obs')
    ax[0][0].plot(modData[sismo].Vz[:, (nStations // 5)], label='mod')
    ax[0][0].set_xlabel('$Samples$', size=14)
    ax[0][0].set_ylabel('$Vz~amplitude$', size=14)
    ax[0][0].set_title('$a)$', loc='center', size=18)    
    ax[0][0].legend()
    
    ax[0][1].plot(obsData[sismo].Vz[:, 2*(nStations // 5)], label='obs')
    ax[0][1].plot(modData[sismo].Vz[:, 2*(nStations // 5)], label='mod')
    ax[0][1].set_xlabel('$Samples$', size=14)
    ax[0][1].set_ylabel('$Vz~amplitude$', size=14)
    ax[0][1].set_title('$b)$', loc='center', size=18)    
    ax[0][1].legend()
    
    ax[1][0].plot(obsData[sismo].Vz[:, 3*(nStations // 5)], label='obs')
    ax[1][0].plot(modData[sismo].Vz[:, 3*(nStations // 5)], label='mod')
    ax[1][0].set_xlabel('$Samples$', size=14)
    ax[1][0].set_ylabel('$Vz~amplitude$', size=14)
    ax[1][0].set_title('$c)$', loc='center', size=18)    
    ax[1][0].legend()
    
    ax[1][1].plot(obsData[sismo].Vz[:, 4*(nStations // 5)], label='obs')
    ax[1][1].plot(modData[sismo].Vz[:, 4*(nStations // 5)], label='mod')
    ax[1][1].set_xlabel('$Samples$', size=14)
    ax[1][1].set_ylabel('$Vz~amplitude$', size=14)
    ax[1][1].set_title('$d)$', loc='center', size=18)    
    ax[1][1].legend()
    
    plt.show()
    fig.savefig(f"../pdf/Actual/Comparacion_obsData_modData_Vz_{fq}Hz.pdf", bbox_inches='tight')



#    f = plt.figure(figsize=(20,10))
#    plt.subplot(2, 2, 1)
#    plt.plot(obsData[sismo].Vz[:, (nStations // 5)], label='obs')
#    plt.plot(modData[sismo].Vz[:, (nStations // 5)], label='mod')
#    plt.xlabel('Muestras', size=14)
#    plt.ylabel('Amplitud', size=14)
#    plt.legend()
#    plt.title(f"Vz Traza {1 * (nStations // 5)}")
#    plt.subplot(2, 2, 2)
#    plt.plot(obsData[sismo].Vz[:, 2 * (nStations // 5)], label='obs')
#    plt.plot(modData[sismo].Vz[:, 2 * (nStations // 5)], label='mod')
#    plt.xlabel('Muestras', size=14)
#    plt.ylabel('Amplitud', size=14)
#    plt.legend()
#    plt.title(f"Vz Traza {2 * (nStations // 5)}")
#    plt.subplot(2, 2, 3)
#    plt.plot(obsData[sismo].Vz[:, 3 * (nStations // 5)], label='obs')
#    plt.plot(modData[sismo].Vz[:, 3 * (nStations // 5)], label='mod')
#    plt.xlabel('Muestras', size=14)
#    plt.ylabel('Amplitud', size=14)
#    plt.legend()
#    plt.title(f"Vz Traza {3 * (nStations // 5)}")
#    plt.subplot(2, 2, 4)
#    plt.plot(obsData[sismo].Vz[:, 4 * (nStations // 5)], label='obs')
#    plt.plot(modData[sismo].Vz[:, 4 * (nStations // 5)], label='mod')
#    plt.xlabel('Muestras', size=14)
#    plt.ylabel('Amplitud', size=14)
#    plt.legend()
#    plt.title(f"Vz Traza {4 * (nStations // 5)}")
#    f.savefig(f"../pdf/Actual/Comparacion_obsData_modData_Vz_{fq}Hz.pdf", bbox_inches='tight')
#    plt.show()

