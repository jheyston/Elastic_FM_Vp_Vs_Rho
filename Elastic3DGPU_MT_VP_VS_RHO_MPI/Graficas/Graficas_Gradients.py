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
    paso = 40
    xticks = np.arange(0, (nx), paso)
    yticks = np.arange(0, (ny), paso)
    zticks = np.arange(0, (nz), paso)
    
    corte_x = corteY
    corte_y = corteY
    corte_z = corteY
    
    xlabel_list = np.arange(0, (nx)*(dx/1e3), paso*(dx/1e3))
    ylabel_list = np.arange(0, (ny)*(dy/1e3), paso*(dy/1e3))
    zlabel_list = np.arange(0, (nz)*(dz/1e3), paso*(dz/1e3))

    
    
    fig, axes = plt.subplots(3,3, figsize=(10, 5), sharex = True, sharey = True)
    

    axes[0][0].imshow(model1[:,:,corte_x], vmin=vmin, vmax=vmax)
    axes[0][0].set_xticks(list(yticks))
    axes[0][0].set_xticklabels(list(ylabel_list.astype(int)))
    axes[0][0].set_yticks(list(zticks))
    axes[0][0].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[0][0].set_ylabel('$Depth (Km)$', size =14)
    axes[0][0].set_xlabel('$Distance (Km)$', size =14)
    axes[0][0].set_title('$a)$', size=14)
    
    axes[0][1].imshow(model1[corte_y,:,:], vmin=vmin, vmax=vmax)
    axes[0][1].set_xticks(list(xticks))
    axes[0][1].set_xticklabels(list(xlabel_list.astype(int)))
    axes[0][1].set_yticks(list(zticks))
    axes[0][1].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[0][1].set_ylabel('$Depth (Km)$', size =14)
    axes[0][1].set_xlabel('$Distance (Km)$', size =14)
    axes[0][1].set_title('$b)$', size=14)
    
    axes[0][2].imshow(model1[:,corte_z, :], vmin=vmin, vmax=vmax)
    axes[0][2].set_xticks(list(xticks))
    axes[0][2].set_xticklabels(list(xlabel_list.astype(int)))
    axes[0][2].set_yticks(list(yticks))
    axes[0][2].set_yticklabels(list(ylabel_list.astype(int)))
   
    axes[0][2].set_ylabel('$Depth (Km)$', size =14)
    axes[0][2].set_xlabel('$Distance (Km)$', size =14)
    axes[0][2].set_title('$c)$', size=14)
    
    axes[1][0].imshow(model2[:,:,corte_x], vmin=vmin, vmax=vmax)
    axes[1][0].set_xticks(list(yticks))
    axes[1][0].set_xticklabels(list(ylabel_list.astype(int)))
    axes[1][0].set_yticks(list(zticks))
    axes[1][0].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[1][0].set_ylabel('$Depth (Km)$', size =14)
    axes[1][0].set_xlabel('$Distance (Km)$', size =14)
    axes[1][0].set_title('$a)$', size=14)
    
    axes[1][1].imshow(model2[corte_y,:,:], vmin=vmin, vmax=vmax)
    axes[1][1].set_xticks(list(xticks))
    axes[1][1].set_xticklabels(list(xlabel_list.astype(int)))
    axes[1][1].set_yticks(list(zticks))
    axes[1][1].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[1][1].set_ylabel('$Depth (Km)$', size =14)
    axes[1][1].set_xlabel('$Distance (Km)$', size =14)
    axes[1][1].set_title('$b)$', size=14)
    
    axes[1][2].imshow(model2[:,corte_z, :], vmin=vmin, vmax=vmax)
    axes[1][2].set_xticks(list(xticks))
    axes[1][2].set_xticklabels(list(xlabel_list.astype(int)))
    axes[1][2].set_yticks(list(yticks))
    axes[1][2].set_yticklabels(list(ylabel_list.astype(int)))
   
    axes[1][2].set_ylabel('$Depth (Km)$', size =14)
    axes[1][2].set_xlabel('$Distance (Km)$', size =14)
    axes[1][2].set_title('$c)$', size=14)
    
    axes[2][0].imshow(model3[:,:,corte_x], vmin=vmin, vmax=vmax)
    axes[2][0].set_xticks(list(yticks))
    axes[2][0].set_xticklabels(list(ylabel_list.astype(int)))
    axes[2][0].set_yticks(list(zticks))
    axes[2][0].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[2][0].set_ylabel('$Depth (Km)$', size =14)
    axes[2][0].set_xlabel('$Distance (Km)$', size =14)
    axes[2][0].set_title('$a)$', size=14)
    
    axes[2][1].imshow(model3[corte_y,:,:], vmin=vmin, vmax=vmax)
    axes[2][1].set_xticks(list(xticks))
    axes[2][1].set_xticklabels(list(xlabel_list.astype(int)))
    axes[2][1].set_yticks(list(zticks))
    axes[2][1].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[2][1].set_ylabel('$Depth (Km)$', size =14)
    axes[2][1].set_xlabel('$Distance (Km)$', size =14)
    axes[2][1].set_title('$b)$', size=14)
    
    axes[2][2].imshow(model3[:,corte_z, :], vmin=vmin, vmax=vmax)
    axes[2][2].set_xticks(list(xticks))
    axes[2][2].set_xticklabels(list(xlabel_list.astype(int)))
    axes[2][2].set_yticks(list(yticks))
    axes[2][2].set_yticklabels(list(ylabel_list.astype(int)))
   
    axes[2][2].set_ylabel('$Depth (Km)$', size =14)
    axes[2][2].set_xlabel('$Distance (Km)$', size =14)
    axes[2][2].set_title('$c)$', size=14)
       
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


def graficarGradients(model1, name1, vmin1, vmax1, model2, name2, vmin2, vmax2, model3, name3, vmin3, vmax3 ):
    im_ratio = model1.shape[0]/model1.shape[1]
    ny, nz, nx = model1.shape
    paso = 40
    xticks = np.arange(0, (nx), paso)
    yticks = np.arange(0, (ny), paso)
    zticks = np.arange(0, (nz), paso)
    
    corte_x = corteY
    corte_y = corteY
    corte_z = corteY
    
    xlabel_list = np.arange(0, (nx)*(dx/1e3), paso*(dx/1e3))
    ylabel_list = np.arange(0, (ny)*(dy/1e3), paso*(dy/1e3))
    zlabel_list = np.arange(0, (nz)*(dz/1e3), paso*(dz/1e3))

    
    
    fig, axes = plt.subplots(3,3, figsize=(12, 10), sharex = True, sharey = True)
    

    axes[0][0].imshow(model1[:,:,corte_x], vmin=vmin1, vmax=vmax1)
    axes[0][0].set_xticks(list(yticks))
    axes[0][0].set_xticklabels(list(ylabel_list.astype(int)))
    axes[0][0].set_yticks(list(zticks))
    axes[0][0].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[0][0].set_ylabel('$Depth (Km)$', size =14)
    axes[0][0].set_xlabel('$Distance (Km)$', size =14)
    axes[0][0].set_title('$a)$', size=14)
    
    axes[0][1].imshow(model1[corte_y,:,:], vmin=vmin1, vmax=vmax1)
    axes[0][1].set_xticks(list(xticks))
    axes[0][1].set_xticklabels(list(xlabel_list.astype(int)))
    axes[0][1].set_yticks(list(zticks))
    axes[0][1].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[0][1].set_ylabel('$Depth (Km)$', size =14)
    axes[0][1].set_xlabel('$Distance (Km)$', size =14)
    axes[0][1].set_title('$b)$', size=14)
    
    axes[0][2].imshow(model1[:,corte_z, :], vmin=vmin1, vmax=vmax1)
    axes[0][2].set_xticks(list(xticks))
    axes[0][2].set_xticklabels(list(xlabel_list.astype(int)))
    axes[0][2].set_yticks(list(yticks))
    axes[0][2].set_yticklabels(list(ylabel_list.astype(int)))
   
    axes[0][2].set_ylabel('$Depth (Km)$', size =14)
    axes[0][2].set_xlabel('$Distance (Km)$', size =14)
    axes[0][2].set_title('$c)$', size=14)
    
    axes[1][0].imshow(model2[:,:,corte_x], vmin=vmin2, vmax=vmax2)
    axes[1][0].set_xticks(list(yticks))
    axes[1][0].set_xticklabels(list(ylabel_list.astype(int)))
    axes[1][0].set_yticks(list(zticks))
    axes[1][0].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[1][0].set_ylabel('$Depth (Km)$', size =14)
    axes[1][0].set_xlabel('$Distance (Km)$', size =14)
    axes[1][0].set_title('$a)$', size=14)
    
    axes[1][1].imshow(model2[corte_y,:,:], vmin=vmin2, vmax=vmax2)
    axes[1][1].set_xticks(list(xticks))
    axes[1][1].set_xticklabels(list(xlabel_list.astype(int)))
    axes[1][1].set_yticks(list(zticks))
    axes[1][1].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[1][1].set_ylabel('$Depth (Km)$', size =14)
    axes[1][1].set_xlabel('$Distance (Km)$', size =14)
    axes[1][1].set_title('$b)$', size=14)
    
    axes[1][2].imshow(model2[:,corte_z, :], vmin=vmin2, vmax=vmax2)
    axes[1][2].set_xticks(list(xticks))
    axes[1][2].set_xticklabels(list(xlabel_list.astype(int)))
    axes[1][2].set_yticks(list(yticks))
    axes[1][2].set_yticklabels(list(ylabel_list.astype(int)))
   
    axes[1][2].set_ylabel('$Depth (Km)$', size =14)
    axes[1][2].set_xlabel('$Distance (Km)$', size =14)
    axes[1][2].set_title('$c)$', size=14)
    
    axes[2][0].imshow(model3[:,:,corte_x], vmin=vmin3, vmax=vmax3)
    axes[2][0].set_xticks(list(yticks))
    axes[2][0].set_xticklabels(list(ylabel_list.astype(int)))
    axes[2][0].set_yticks(list(zticks))
    axes[2][0].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[2][0].set_ylabel('$Depth (Km)$', size =14)
    axes[2][0].set_xlabel('$Distance (Km)$', size =14)
    axes[2][0].set_title('$a)$', size=14)
    
    axes[2][1].imshow(model3[corte_y,:,:], vmin=vmin3, vmax=vmax3)
    axes[2][1].set_xticks(list(xticks))
    axes[2][1].set_xticklabels(list(xlabel_list.astype(int)))
    axes[2][1].set_yticks(list(zticks))
    axes[2][1].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[2][1].set_ylabel('$Depth (Km)$', size =14)
    axes[2][1].set_xlabel('$Distance (Km)$', size =14)
    axes[2][1].set_title('$b)$', size=14)
    
    axes[2][2].imshow(model3[:,corte_z, :], vmin=vmin3, vmax=vmax3)
    axes[2][2].set_xticks(list(xticks))
    axes[2][2].set_xticklabels(list(xlabel_list.astype(int)))
    axes[2][2].set_yticks(list(yticks))
    axes[2][2].set_yticklabels(list(ylabel_list.astype(int)))
   
    axes[2][2].set_ylabel('$Depth (Km)$', size =14)
    axes[2][2].set_xlabel('$Distance (Km)$', size =14)
    axes[2][2].set_title('$c)$', size=14)
    
        
    plt.show()
    fig.savefig(f"../pdf/Actual/Gradients_{name1}_and_{name2}_{name3}.pdf", bbox_inches='tight')
    
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
    # -----------------------------------------------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#
#    path = "FWI_LambdaCte_Mu_RhoCte_Seismology"
#    path = 'FWI_Lambda_Mu_RhoCte_Seismology'
#    path = 'FWI_Lambda_Mu_Rho_Seismology_Nsrc25'
    path = "Actual"
#    path = "FWI_Lambda_Mu_Rho_Seismology_Nsrc25_beta4"
#    path = "FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite"

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
    
    gMu_full = load(f'../Results/{path}/Gradients/gMu_{fq}Hz_full.npy')
    gLambda_full = load(f'../Results/{path}/Gradients/gLambda_{fq}Hz_full.npy')
    gRho_full = load(f'../Results/{path}/Gradients/gRho_{fq}Hz_full.npy')
    
    gMu_temporal = load(f'../Results/{path}/Gradients/gMu_{fq}Hz_temporal.npy')
    gLambda_temporal = load(f'../Results/{path}/Gradients/gLambda_{fq}Hz_temporal.npy')
    gRho_temporal = load(f'../Results/{path}/Gradients/gRho_{fq}Hz_temporal.npy')
    
    gMu_SGC = load(f'../Results/{path}/Gradients/gMu_{fq}Hz_SGC.npy')
    gLambda_SGC = load(f'../Results/{path}/Gradients/gLambda_{fq}Hz_SGC.npy')
    gRho_SGC = load(f'../Results/{path}/Gradients/gRho_{fq}Hz_SGC.npy')
    
    gMu_SGC_temporal_permanente = load(f'../Results/{path}/Gradients/gMu_{fq}Hz_SGC_temporal_permanente.npy')
    gLambda_SGC_temporal_permanente = load(f'../Results/{path}/Gradients/gLambda_{fq}Hz_SGC_temporal_permanente.npy')
    gRho_SGC_temporal_permanente = load(f'../Results/{path}/Gradients/gRho_{fq}Hz_SGC_temporal_permanente.npy')

    # -----------------------------------------------------------------------------------------------------------------#
    ny, nz, nx = VpTarget.shape
#    plt.figure()
#    plt.imshow(obsData[1].Vz, aspect='auto')
#    plt.show()
    
    paso = 40
    xticks = np.arange(0, (nx), paso)
    zticks = np.arange(0, -(nz), -paso)

    xlabel_list = np.arange(0, (nx)*(dx/1e3), paso*(dx/1e3))
    zlabel_list = np.arange(0, (nz)*(dz/1e3), paso*(dz/1e3))

    
    f = plt.figure()
    figax = plt.axes(projection='3d')
    figax.scatter3D(sta_pos_x, sta_pos_y, -sta_pos_z, cmap='Greens', label='Estaciones')
    figax.scatter3D(sismo_pos_x, sismo_pos_y, -sismo_pos_z, cmap='Greens', label='Sismos')
    figax.set_xlabel('X', size=18)
    figax.set_ylabel('Y', size=18)
    figax.set_zlabel('Z', size=18)
    figax.legend()
    figax.set(xlim=(0, nx), ylim=(0, ny), zlim=(-nz, 0))

    plt.show(block=False)
    plt.pause(2.0)
    plt.close()
    # plt.set(xlim=(0, nx), ylim=(-nz, 0))
    f.savefig(f"../pdf/Actual/Geometry_seismology.pdf", bbox_inches='tight')
    # -----------------------------------------------------------------------------------------------------------------#

    nSismos = len(obsData)
    nt, nStations = modData[sismo].Vx.shape

    print(nt, nStations)
    

    
    vmingMu = np.amin(gMu_SGC)*1e-1
    vmaxgMu = np.amax(gMu_SGC)*1e-1
    
    vmingLambda = np.amin(gLambda_SGC)*1e-1
    vmaxgLambda = np.amax(gLambda_SGC)*1e-1
    
    vmingRho = np.amin(gRho_SGC)*1e-1
    vmaxgRho = np.amax(gRho_SGC)*1e-1
    
    Vs = np.sqrt((Mu)/Rho)
    Vp = np.sqrt((Lambda+2*Mu)/Rho)

    graficarGradients(gLambda_full, 'gLambda_full',  vmingLambda, vmaxgLambda, gRho_full, 'gRho_full', vmingRho, vmaxgRho,   gMu_full, 'gMu_full',   vmingMu, vmaxgMu)
    graficarGradients(gLambda_temporal, 'gLambda_temporal',  vmingLambda, vmaxgLambda, gRho_temporal, 'gRho_temporal', vmingRho, vmaxgRho,   gMu_temporal, 'gMu_temporal',   vmingMu, vmaxgMu)
    graficarGradients(gLambda_SGC, 'gLambda_SGC',  vmingLambda, vmaxgLambda, gRho_SGC, 'gRho_SGC', vmingRho, vmaxgRho,   gMu_SGC, 'gMu_SGC',   vmingMu, vmaxgMu)
    graficarGradients(gLambda_SGC_temporal_permanente, 'gLambda_SGC_temporal_permanente',  vmingLambda, vmaxgLambda, gRho_SGC_temporal_permanente, 'gRho_SGC_temporal_permanente', vmingRho, vmaxgRho,   gMu_SGC_temporal_permanente, 'gMu_SGC_temporal_permanente',   vmingMu, vmaxgMu)
    
    
    
