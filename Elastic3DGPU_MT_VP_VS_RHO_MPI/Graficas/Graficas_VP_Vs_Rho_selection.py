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
    fq = 0.3
    dx = 1e3
    dz = dx
    sismo = 5
    # -----------------------------------------------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#
    pathList = []
    pathList.append( "FWI_Lambda_Mu_RhoCteNafeDrake_Seismology_Nsrc25")
    pathList.append( "FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite")
    pathList.append( "FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_2000ite")
    


    
    titles = ['Initial',  'RhoCteNafeDrake',  'RhoNafeDrake TM_est', 'RhoNafeDrake TM_est 2000ite']

    
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
    
    for it in range(nModels):
        Rho[:,:,:,it] = load(f'../Results/{pathList[it]}/Models/Rho_{fq}Hz.npy')
        Lambda[:,:,:,it] = load(f'../Results/{pathList[it]}/Models/Lambda_{fq}Hz.npy')
        Mu[:,:,:,it] = load(f'../Results/{pathList[it]}/Models/Mu_{fq}Hz.npy')
        
        for j in range(ny):
                for k in range(nz):
                    for i in range(nx):
                        if Mu[j, k, i, it]< 0:
                            Mu[j, k, i, it] = (Mu[j, k, i+1, it]+Mu[j, k, i-1, it]+Mu[j, k+1, i, it]+Mu[j, k-1, i, it])/4
        Vs[:,:,:,it] = np.sqrt((Mu[:,:,:,it])/Rho[:,:,:,it])
        Vp[:,:,:,it] = np.sqrt((Lambda[:,:,:,it]+2*Mu[:,:,:,it])/Rho[:,:,:,it])
        
   

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
    plt.show()
    
    wCPML = 22
    
    RhoInicial = RhoInicial[wCPML:ny-wCPML,  0:nz-wCPML, wCPML:nx-wCPML].copy()
    VpInicial = VpInicial[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML].copy()
    VsInicial = VsInicial[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML].copy()
    RhoTarget = RhoTarget[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML].copy()
    VsTarget = VsTarget[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML].copy()
    VpTarget = VpTarget[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML].copy()
    
    Vp = Vp[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML, :].copy()
    Vs = Vs[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML, :].copy()
    Rho = Rho[wCPML:ny-wCPML, 0:nz-wCPML, wCPML:nx-wCPML, :].copy()
    
    print(Vp.shape)
    print(Vs.shape)
    print(Rho.shape)



    beta_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    errorModelRho = [np.sqrt(1/((nx-2*wCPML)*(ny-2*wCPML)*(nz-2*wCPML))* np.sum(((RhoInicial.ravel()-RhoTarget.ravel() )/RhoTarget.ravel())**2) )]
    errorModelVp = [np.sqrt(1/((nx-2*wCPML)*(ny-2*wCPML)*(nz-2*wCPML)) * np.sum(((VpInicial.ravel()-VpTarget.ravel() )/VpTarget.ravel())**2) ) ]
    errorModelVs = [np.sqrt(1/((nx-2*wCPML)*(ny-2*wCPML)*(nz-2*wCPML)) * np.sum(((VsInicial.ravel()-VsTarget.ravel() )/VsTarget.ravel())**2) ) ]
    for i in range(nModels):
        errorModelRho.append(np.sqrt(1/((nx-2*wCPML)*(ny-2*wCPML)*(nz-2*wCPML))* np.sum(((Rho[:,:,:,i].ravel()-RhoTarget.ravel() )/RhoTarget.ravel())**2) ) )
        errorModelVs.append(np.sqrt(1/((nx-2*wCPML)*(ny-2*wCPML)*(nz-2*wCPML))* np.sum(((Vs[:,:,:,i].ravel()-VsTarget.ravel() )/VsTarget.ravel())**2) ) )
        errorModelVp.append(np.sqrt(1/((nx-2*wCPML)*(ny-2*wCPML)*(nz-2*wCPML))* np.sum(((Vp[:,:,:,i].ravel()-VpTarget.ravel() )/VpTarget.ravel())**2) ) )
        
    
    for i in range(nModels+1):
        print(f"{titles [i]}  % error Rho: {errorModelRho[i]:.4f} % error Vp: {errorModelVp[i]:.4f} % error Vs: {errorModelVs[i]:.4f}")

    
    paso = 40
    corteY = 30
    corteX = [10, 30, 50]
    xticks = np.arange(0, (nx), paso)
    zticks = np.arange(0, -(nz), -paso)

    xlabel_list = np.arange(0, (nx)*(dx/1e3), paso*(dx/1e3))
    zlabel_list = np.arange(0, (nz)*(dz/1e3), paso*(dz/1e3))

#    f = plt.figure()
    fig, ax = plt.subplots(3, 3, sharex = True, figsize=(20,10))
    zticks = np.arange(0, (nz), paso)
    ax[0][0].plot(VpTarget[corteY, :,  corteX[0]], label='$Real~ Vp$')
    ax[0][0].plot(VpInicial[corteY, :,  corteX[0]], label='$Initial~ Vp$')
    
    ax[1][0].plot(VpTarget[corteY, :,  corteX[1]], label='$Real~ Vs$')
    ax[1][0].plot(VpInicial[corteY, :,  corteX[1]], label='$Initial~ Vs$')

    ax[2][0].plot(VpTarget[corteY, :,  corteX[2]], label='$Real~ Vp$')
    ax[2][0].plot(VpInicial[corteY, :, corteX[2]], label='$Initial~ Vp$')
    
    ax[0][1].plot(VsTarget[corteY, :,  corteX[0]], label='$Real~ Vs$')
    ax[0][1].plot(VsInicial[corteY, :,  corteX[0]], label='$Initial~ Vs$')
    
    ax[1][1].plot(VsTarget[corteY, :,  corteX[1]], label='$Real~ Vs$')
    ax[1][1].plot(VsInicial[corteY, :,  corteX[1]], label='$Initial~ Vs$')
    
    ax[2][1].plot(VsTarget[corteY, :,  corteX[2]], label='$Real~ Vs$')
    ax[2][1].plot(VsInicial[ corteY, :,  corteX[2]], label='$Initial~ Vs$')
    
    ax[0][2].plot(RhoTarget[corteY, :, corteX[0]], label=r'$Real~ \rho$')
    ax[0][2].plot(RhoInicial[corteY, :,  corteX[0]], label=r'$Initial~ \rho$')

    ax[1][2].plot(RhoTarget[corteY,  :, corteX[1]], label=r'$Real~ \rho$')
    ax[1][2].plot(RhoInicial[corteY, :, corteX[1]], label=r'$Initial~ \rho$')    
    
    ax[2][2].plot(RhoTarget[corteY, :, corteX[2]], label=r'$Real~ \rho$')
    ax[2][2].plot(RhoInicial[corteY, :,  corteX[2]], label=r'$Initial~ \rho$')
    for i in range(0, nModels):
    
        ax[0][0].plot(Vp[corteY, :,  corteX[0], i], label=f'Vp-{titles[i+1]}')
        ax[0][0].set_ylabel('$Vp(m/s)$', size=14)
        ax[0][0].set_xlabel('$Depth (km)$', size=14)
        ax[0][0].set_title('$a)$', loc='left', size=18)    
        ax[0][0].legend(loc='lower right')
        ax[0][0].set_xticks(list(zticks))
        ax[0][0].set_xticklabels(list(zlabel_list.astype(int)))
        ax[0][0].set_ylim([1200, 5000])
        
        ax[1][0].plot(Vp[corteY, :, corteX[1], i], label=f'Vp-{titles[i+1]}')
        ax[1][0].set_ylabel('$Vp(m/s)$', size=14)
        ax[1][0].set_xlabel('$Depth (km)$', size=14)
        ax[1][0].set_title('$d)$', loc='left', size=18)  
        ax[1][0].set_xticks(list(zticks))
        ax[1][0].set_xticklabels(list(zlabel_list.astype(int)))
        ax[1][0].legend(loc='lower right')
        ax[1][0].set_ylim([1200, 5000])
    
        ax[2][0].plot(Vp[corteY, :,  corteX[2], i],label=f'Vp-{titles[i+1]}')
        ax[2][0].set_ylabel('$Vp(m/s)$', size=14)
        ax[2][0].set_xlabel('$Depth (km)$', size=14)
        ax[2][0].set_title('$g)$', loc='left', size=18)    
        ax[2][0].set_xticks(list(zticks))
        ax[2][0].set_xticklabels(list(zlabel_list.astype(int)))
        ax[2][0].legend(loc='lower right')
        ax[2][0].set_ylim([1200, 5000])
    
        
        ax[0][1].plot(Vs[corteY, :,  corteX[0], i], label=f'Vs-{titles[i+1]}')
        ax[0][1].set_ylabel('$Vs(m/s)$', size=14)
        ax[0][1].set_xlabel('$Depth (km)$', size=14)
        ax[0][1].set_title('$b)$', loc='left', size=18)    
        ax[0][1].legend(loc='lower right')
        ax[0][1].set_ylim([600, 2600])
        
        ax[1][1].plot(Vs[corteY, :,  corteX[1], i], label=f'Vs-{titles[i+1]}')
        ax[1][1].set_ylabel('$Vs(m/s)$', size=14)
        ax[1][1].set_xlabel('$Depth (km)$', size=14)
        ax[1][1].set_title('$e)$', loc='left', size=18)    
        ax[1][1].legend(loc='lower right')
        ax[1][1].set_ylim([600, 2600])
        
        ax[2][1].plot(Vs[corteY, :,  corteX[2], i], label=f'Vs-{titles[i+1]}')
        ax[2][1].set_ylabel('$Vs(m/s)$', size=14)
        ax[2][1].set_xlabel('$Depth (km)$', size=14)
        ax[2][1].set_title('$h)$', loc='left', size=18)    
        ax[2][1].set_xticks(list(zticks))
        ax[2][1].set_xticklabels(list(zlabel_list.astype(int)))
        ax[2][1].legend(loc='lower right')
        ax[2][1].set_ylim([600, 2600])
    
        ax[0][2].plot(Rho[corteY, :,  corteX[0], i], label=f'Rho-{titles[i+1]}')
        ax[0][2].set_ylabel(r'$\rho (kg/m^3)$', size=14)
        ax[0][2].set_xlabel('$Depth (km)$', size=14)
        ax[0][2].set_title('$c)$', loc='left', size=18)    
        ax[0][2].legend(loc='lower right')
        ax[0][2].set_ylim([1300, 3000])
    
        
        ax[1][2].plot(Rho[ corteY, :, corteX[1], i], label=f'Rho-{titles[i+1]}')
        ax[1][2].set_ylabel(r'$\rho (kg/m^3)$', size=14)
        ax[1][2].set_xlabel('$Depth (km)$', size=14)
        ax[1][2].set_title('$f)$', loc='left', size=18)    
        ax[1][2].legend(loc='lower right')
        ax[1][2].set_ylim([1300, 3000])
    
        
        ax[2][2].plot(Rho[corteY, :,  corteX[2], i], label=f'Rho-{titles[i+1]}')
        ax[2][2].set_ylabel(r'$\rho (kg/m^3)$', size=14)
        ax[2][2].set_xlabel('$Depth (km)$', size=14)
        ax[2][2].set_title('$i)$', loc='left', size=18)   
        ax[2][2].set_xticks(list(zticks))
        ax[2][2].set_xticklabels(list(zlabel_list.astype(int)))
        ax[2][2].legend(loc='lower right')
        ax[2][2].set_ylim([1300, 3000])
       
#        fig.savefig(f"../pdf/Actual/corteVp_Vs_{fq}Hz.pdf", bbox_inches='tight')
    
            
#    
#    f = plt.figure()
#    plt.rc('xtick', labelsize=20) 
#    plt.rc('ytick', labelsize=20) 
#    plt.plot(beta_values, errorModelRho[1:], marker='.', markersize=14, label=r'$\rho$')
#    plt.plot(beta_values, errorModelVp[1:], marker='.', markersize=14, label=r'$Vp$')
#    plt.plot(beta_values, errorModelVs[1:], marker='.', markersize=14, label=r'$Vs$')
#    for i in range(len(beta_values)):
#        plt.text(beta_values[i], errorModelRho[i+1]+0.002, f"${errorModelRho[i+1]:.2}$", size=14)
#        plt.text(beta_values[i], errorModelVp[i+1]-0.002, f"${errorModelVp[i+1]:.2}$", size=14)
#        plt.text(beta_values[i], errorModelVs[i+1]+0.001, f"${errorModelVs[i+1]:.2}$", size=14)
#
#    plt.ylabel(r'$\% ~error$', size=20)
#    plt.xlabel(r'$1/\beta$', size=20)
#    plt.legend(prop={"size":16})
#
#
#    plt.show()
#
#    f.savefig(f"../pdf/porcentaje_error_beta.pdf", bbox_inches='tight')
    