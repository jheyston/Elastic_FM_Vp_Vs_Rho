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
    nz, nx = model1.shape
    paso = 40
    xticks = np.arange(0, (nx), paso)
    zticks = np.arange(0, (nz), paso)

    xlabel_list = np.arange(0, (nx)*(dx/1e3), paso*(dx/1e3))
    zlabel_list = np.arange(0, (nz)*(dz/1e3), paso*(dz/1e3))

    
    
    fig, axes = plt.subplots(1,3, figsize=(10, 5), sharex = True, sharey = True)
    

    im = axes[0].imshow(model1, vmin=vmin, vmax=vmax)
    axes[0].set_xticks(list(xticks))
    axes[0].set_xticklabels(list(xlabel_list.astype(int)))
    axes[0].set_yticks(list(zticks))
    axes[0].set_yticklabels(list(zlabel_list.astype(int)))
   
    axes[0].set_ylabel('$Depth (Km)$', size =14)
    axes[0].set_xlabel('$Distance (Km)$', size =14)
    axes[0].set_title('$a)$', size=14)
    
       
    im = axes[1].imshow(model2, vmin=vmin, vmax=vmax)
    axes[1].set_xticks(list(xticks))
    axes[1].set_xticklabels(list(xlabel_list.astype(int)))
    axes[1].set_yticks(list(zticks))
    axes[1].set_yticklabels(list(zlabel_list.astype(int)))
    
    axes[1].set_ylabel('$Depth (Km)$', size =14)
    axes[1].set_xlabel('$Distance (Km)$', size =14)
    axes[1].set_title('$b)$', size=14)
    
    
    im = axes[2].imshow(model3, vmin=vmin, vmax=vmax)
    
    axes[2].set_xticks(list(xticks))
    axes[2].set_xticklabels(list(xlabel_list.astype(int)))
    axes[2].set_yticks(list(zticks))
    axes[2].set_yticklabels(list(zlabel_list.astype(int)))
    
    axes[2].set_ylabel('$Depth (Km)$', size =14)
    axes[2].set_xlabel('$Distance (Km)$', size =14)
    axes[2].set_title('$c)$', size=14)
    plt.colorbar(im, fraction=0.047*im_ratio)
    

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
    # -----------------------------------------------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#
    pathList = []
    pathList.append("FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite_sigma0")
    pathList.append("FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite")
    pathList.append("FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite_sigma2")
    pathList.append("FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite_sigma3")
    pathList.append("FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite_sigma4")
    pathList.append("FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite_sigma5")
    pathList.append("FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite_sigma6")
    pathList.append("FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite_sigma7")

    
    
    
    path = "Actual"
    ''' Carga de archivos '''
    ''' Target '''
    VpTarget = load(f'../Results/{path}/Models/VpTarget.npy')
    VsTarget = load(f'../Results/{path}/Models/VsTarget.npy')
    RhoTarget = load(f'../Results/{path}/Models/RhoTarget.npy')
    LambdaTarget = load(f'../Results/{path}/Models/LambdaTarget.npy')
    MuTarget = load(f'../Results/{path}/Models/MuTarget.npy')
    ''' Initial '''
    VpInicial = load(f'../Results/{path}/Models/VpInicial.npy')
    VsInicial = load(f'../Results/{path}/Models/VsInicial.npy')
    RhoInicial = load(f'../Results/{path}/Models/RhoInicial.npy')
    LambdaInicial = load(f'../Results/{path}/Models/LambdaInicial.npy')
    MuInicial = load(f'../Results/{path}/Models/MuInicial.npy')
    
    costFunction = load(f'../Results/{path}/CostFunction/costFunction_{fq}Hz.npy')
    
    ny, nz, nx = VpTarget.shape
    
    ''' path 1 '''
    nModels = len(pathList)
    Rho = np.zeros([ny, nz, nx, nModels])
    Lambda = np.zeros([ny, nz, nx, nModels])
    Mu = np.zeros([ny, nz, nx, nModels])
    Vs = np.zeros([ny, nz, nx, nModels])
    Vp = np.zeros([ny, nz, nx, nModels])
    
    for i in range(nModels):
        
        tempRho = load(f'../Results/{pathList[i]}/Models/Rho_{fq}Hz.npy')
        tempLambda = load(f'../Results/{pathList[i]}/Models/Lambda_{fq}Hz.npy')
        tempMu = load(f'../Results/{pathList[i]}/Models/Mu_{fq}Hz.npy')
            
        
        Rho[:,:, :,i] = tempRho
        Lambda[:,:,:,i] = tempLambda
        Mu[:,:, :,i] = tempMu
        
        print(np.amin(Rho[:,:, :,i]), np.amin(Mu[:,:, :,i]))
        
        for iy in range(ny):
            for iz in range(nz):
                for ix in range(nx):
                    if Mu[iy, iz, ix,i]< 0:
 
                        Mu[iy,iz, ix, i] = MuInicial[iy,iz, ix] 

                        
        tempVs = np.sqrt((Mu[:,:,:,i])/Rho[:,:,:,i])
        tempVp = np.sqrt((Lambda[:,:,:,i]+2*Mu[:,:,:,i])/Rho[:,:,:,i])
        

#        indVs = np.isnan(tempVs)
#        indVp = np.isnan(tempVp)

        Vs[:,:,:,i] = tempVs
        Vp[:,:,:,i] = tempVp
        
#        for iy in range(ny):
#            for iz in range(nz):
#                for ix in range(nx):
#                    if np.isnan(tempVs[iy, iz, ix]):
# 
#                        Vs[iy,iz, ix, i] = VsInicial[iy,iz, ix] 
#                        Vp[iy,iz, ix, i] = VpInicial[iy,iz, ix] 
 
            
        
        
    
 
    wCPML = 20
    
    RhoInicial = RhoInicial[wCPML:ny-wCPML,  0:nz-wCPML, wCPML:nx-wCPML].copy()
    VpInicial = VpInicial[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML].copy()
    VsInicial = VsInicial[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML].copy()
    RhoTarget = RhoTarget[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML].copy()
    VsTarget = VsTarget[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML].copy()
    VpTarget = VpTarget[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML].copy()
    
    Vp = Vp[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML, :].copy()
    Vs = Vs[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML, :].copy()
    Rho = Rho[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML, :].copy()

    # -----------------------------------------------------------------------------------------------------------------#

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

#    graficarModel(VsTarget, 'VsTarget', VsInicial, 'VsInicial', Vs[:,:,1], 'Vs',   fq, vminVs, vmaxVs)
#    graficarModel(VpTarget, 'VpTarget',  VpInicial, 'VpInicial', np.sqrt((Lambda+2*Mu)/Rho), 'Vp',  fq, vminVp, vmaxVp)
#    graficarModel(RhoTarget, 'RhoTarget', RhoInicial, 'RhoInicial', Rho[:,:,2], 'Rho',  fq, vminRho, vmaxRho)
#    graficarModel(RhoTarget, 'RhoTarget', RhoInicial, 'RhoInicial', Rho[:,:,3], 'Rho',  fq, vminRho, vmaxRho)
#    graficarModel(LambdaTarget, 'LambdaTarget', LambdaInicial, 'LambdaInicial', Lambda, 'Lambda',  fq, vminLambda, vmaxLambda)
#    graficarModel(MuTarget, 'MuTarget', MuInicial,  'MuInicial', Mu, 'Mu', fq, vminMu, vmaxMu)
#    plt.show()
    
    
    errorModelRho = [np.sqrt(1/((nx-2*wCPML)*(ny-2*wCPML)*(nz-2*wCPML))* np.sum(((RhoInicial.ravel()-RhoTarget.ravel() )/RhoTarget.ravel())**2) )]
    errorModelVp = [np.sqrt(1/((nx-2*wCPML)*(ny-2*wCPML)*(nz-2*wCPML)) * np.sum(((VpInicial.ravel()-VpTarget.ravel() )/VpTarget.ravel())**2) ) ]
    errorModelVs = [np.sqrt(1/((nx-2*wCPML)*(ny-2*wCPML)*(nz-2*wCPML)) * np.sum(((VsInicial.ravel()-VsTarget.ravel() )/VsTarget.ravel())**2) ) ]
    for i in range(nModels):
        errorModelRho.append(np.sqrt(1/((nx-2*wCPML)*(ny-2*wCPML)*(nz-2*wCPML))* np.sum(((Rho[:,:,:,i].ravel()-RhoTarget.ravel() )/RhoTarget.ravel())**2) ) )
        errorModelVs.append(np.sqrt(1/((nx-2*wCPML)*(ny-2*wCPML)*(nz-2*wCPML))* np.sum(((Vs[:,:,:,i].ravel()-VsTarget.ravel() )/VsTarget.ravel())**2) ) )
        errorModelVp.append(np.sqrt(1/((nx-2*wCPML)*(ny-2*wCPML)*(nz-2*wCPML))* np.sum(((Vp[:,:,:,i].ravel()-VpTarget.ravel() )/VpTarget.ravel())**2) ) )
        
    
    print(f"% error Rho: {errorModelRho}")
    print(f"% error Vp: {errorModelVp}")
    print(f"% error Vs: {errorModelVs}")
        
    sigma_values = [x for x in range(nModels)]
    f = plt.figure()
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    plt.plot(sigma_values, errorModelRho[1:], marker='.', markersize=14, label=r'$\rho$')
    plt.plot(sigma_values, errorModelVp[1:], marker='.', markersize=14, label=r'$Vp$')
    plt.plot(sigma_values, errorModelVs[1:], marker='.', markersize=14, label=r'$Vs$')
    
    for i in range(len(sigma_values)):
        plt.text(sigma_values[i]+0.03, errorModelRho[i+1]-0.001, f"${errorModelRho[i+1]:.2}$", size=14)
        plt.text(sigma_values[i]+0.01, errorModelVp[i+1]+0.001, f"${errorModelVp[i+1]:.2}$", size=14)
        plt.text(sigma_values[i], errorModelVs[i+1]+0.003, f"${errorModelVs[i+1]:.2}$", size=14)
    plt.ylabel(r'$\% ~error$', size=20)
    plt.xlabel(r'$\sigma$', size=20)
    plt.legend(prop={"size":16})
    plt.show()
    f.savefig(f"../pdf/porcentaje_error_sigma_3D.pdf", bbox_inches='tight')
    
    