"""
Created on Tue Aug 24 12:57:45 2021

@author: jheyston
"""
from lbfgs import compute_sk_yk, L_BFGS
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from numpy import save, load
import matplotlib.pyplot as plt
from math import pi, sqrt, exp, fabs
from tqdm import tqdm
import os
import imageio as imageio
from numba import jit
import time
from scipy.linalg import norm
import copy 
from mpi4py import MPI
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------------------------------------------------#
'''  MPI modules '''
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
sizeRank = comm.Get_size()
# ---------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------------------------- #
''' Initialize Drive PyCuda '''
drv.init()  
dev = drv.Device(rank)
''' Initialize Context PyCuda  '''
ctx = dev.make_context()
ctx.push() 
# -------------------------------------------------------------------------------------------------------------------- #
'''------ CUDA functions ---- '''

ker = SourceModule("""
#define idx  ( threadIdx.x + blockIdx.x * blockDim.x )
#define idy  ( threadIdx.y + blockIdx.y * blockDim.y )
#define idz  ( threadIdx.z + blockIdx.z * blockDim.z )

#define WIDTH  ( blockDim.x * gridDim.x )
#define HEIGHT ( blockDim.y * gridDim.y )

#define XM(x)  ( (x + WIDTH) % WIDTH )
#define YM(y)  ( (y + HEIGHT) % HEIGHT )

#define J(n1, x, y)   ( XM(x)  + YM(y) * n1 )
#define H_STENCIL 2




__global__ void kernel_Tau( float *Tauxx, float *Tauyy, float *Tauzz, 
                            float *Tauxy, float *Tauxz, float *Tauyz,
                            float *Vx, float *Vy,  float *Vz,
                            float *psidVx_dx, float *psidVy_dy, float *psidVz_dz,
                            float *psidVx_dz, float *psidVx_dy, float *psidVy_dx,
                            float *psidVy_dz, float *psidVz_dx, float *psidVz_dy,
                            float *ax, float *bx, float *ay, float *by, float *az, float *bz,
                            float *vp, float *vs, float *rho, float *lambda, float *mu,
                            float dx, float dy, float dz, float dt,
                            int nx, int ny, int nz){
    int x = idx;
    int y = idy;
    int z = idz;
    int tid = idx + idz * nx + idy * nz * nx;

    float Lambda, Mu, Mu_i, Mu_j, Mu_k, Mu_ik, Mu_jk, Mu_ij,  Mu_xy, Mu_xz, Mu_yz;
    float dVx_dx, dVy_dy, dVz_dz;
    float dVx_dz, dVx_dy, dVy_dx, dVy_dz, dVz_dx, dVz_dy;
    if(x>=H_STENCIL && x<nx-H_STENCIL && z>=H_STENCIL && z< nz-H_STENCIL && y>=H_STENCIL && y< ny - H_STENCIL){
        Mu =     mu[tid];
        Lambda = lambda[tid];
        Mu_i  = mu[tid+1];
        Mu_k  = mu[tid+1*nx];
        Mu_j  = mu[tid+1*nz*nx];
        Mu_ik = mu[tid+1+1*nx];
        Mu_jk = mu[tid+1*nx+1*nz*nx];
        Mu_ij = mu[tid+1+1*nz*nx];

        Mu_xz=   1/(0.25/Mu+0.25/Mu_i+0.25/Mu_k+0.25/Mu_ik);
        Mu_xy=   1/(0.25/Mu+0.25/Mu_i+0.25/Mu_j+0.25/Mu_ij);
        Mu_yz=   1/(0.25/Mu+0.25/Mu_j+0.25/Mu_k+0.25/Mu_jk);
        
        dVx_dx = Vx[tid] - Vx[tid-1];
        dVy_dy = Vy[tid] - Vy[tid-1*nz*nx];
        dVz_dz = Vz[tid] - Vz[tid-1*nx];
        
        dVx_dz = Vx[tid+1*nx] - Vx[tid];
        dVx_dy = Vx[tid+1*nx*nz] - Vx[tid];
        dVy_dx = Vy[tid+1] - Vy[tid];
        dVy_dz = Vy[tid+1*nx] - Vy[tid];
        dVz_dx = Vz[tid+1]-Vz[tid];
        dVz_dy = Vz[tid+1*nz*nx]-Vz[tid];
       
        psidVx_dx[tid] = bx[x]*psidVx_dx[tid] + ax[x]*dVx_dx;
        psidVy_dx[tid] = bx[x]*psidVy_dx[tid] + ax[x]*dVy_dx;
        psidVz_dx[tid] = bx[x]*psidVz_dx[tid] + ax[x]*dVz_dx;
        
        psidVy_dy[tid] = by[y]*psidVy_dy[tid] + ay[y]*dVy_dy;
        psidVx_dy[tid] = by[y]*psidVx_dy[tid] + ay[y]*dVx_dy;
        psidVz_dy[tid] = by[y]*psidVz_dy[tid] + ay[y]*dVz_dy;
        
        psidVz_dz[tid] = bz[z]*psidVz_dz[tid] + az[z]*dVz_dz;        
        psidVx_dz[tid] = bz[z]*psidVx_dz[tid] + az[z]*dVx_dz;
        psidVy_dz[tid] = bz[z]*psidVy_dz[tid] + az[z]*dVy_dz;
        
        Tauxx[tid] = Tauxx[tid]+ (dt/dx)*((Lambda+2.0*Mu)*(dVx_dx+psidVx_dx[tid]) 
                     + Lambda * (dVy_dy+dVz_dz+psidVy_dy[tid] + psidVz_dz[tid]) );
                     
        Tauyy[tid] = Tauyy[tid]+ (dt/dx)*((Lambda+2.0*Mu)*(dVy_dy+psidVy_dy[tid]) 
                     + Lambda * (dVx_dx+dVz_dz+psidVx_dx[tid]+ psidVz_dz[tid]) );
                     
        Tauzz[tid] = Tauzz[tid]+ (dt/dx)*((Lambda+2.0*Mu)*(dVz_dz+psidVz_dz[tid]) 
                     + Lambda * (dVx_dx+dVy_dy +psidVx_dx[tid]+ psidVy_dy[tid]) );
                     
        Tauxy[tid] =Tauxy[tid]+ (dt/dx)*Mu_xy*( (dVx_dy + psidVx_dy[tid]) + (dVy_dx + psidVy_dx[tid]) );
        Tauxz[tid] =Tauxz[tid]+ (dt/dx)*Mu_xz*( (dVx_dz + psidVx_dz[tid]) + (dVz_dx + psidVz_dx[tid]) );
        Tauyz[tid] =Tauyz[tid]+ (dt/dx)*Mu_yz*( (dVy_dz + psidVy_dz[tid]) + (dVz_dy + psidVz_dy[tid]) );

    }
}


                               
                                
__global__ void kernel_Vx_Vy_Vz(float *Vx, float *Vy, float *Vz, 
                                float *Tauxx, float *Tauyy, float *Tauzz, 
                                float *Tauxy, float *Tauxz, float *Tauyz,
                                float *psidTauxx_dx, float *psidTauzz_dz, float *psidTauyy_dy,
                                float *psidTauxy_dy, float *psidTauxy_dx, float *psidTauxz_dz,
                                float *psidTauxz_dx, float *psidTauyz_dz, float *psidTauyz_dy,
                                float *ax, float *bx, float *ay, float *by, float *az, float *bz,
                                float *rho,
                                float dx, float dy, float dz, float dt,
                                int nx, int ny, int nz){
    int x = idx;
    int y = idy;
    int z = idz;
    int tid = idx + idz * nx + idy * nz * nx;
    
    float  b, b_i, b_j, b_k, b_x, b_y, b_z;
    float dTauxx_dx, dTauxy_dy, dTauxy_dx, dTauxz_dz;
    float dTauxz_dx, dTauzz_dz, dTauyz_dy, dTauyz_dz, dTauyy_dy;
    
    if(x>=H_STENCIL && x<nx-H_STENCIL && z>=H_STENCIL && z<nz-H_STENCIL && y>=H_STENCIL && y<ny-H_STENCIL){
        b     =   1.0/rho[tid];
        b_i   =   1.0/rho[tid+1];
        b_k   =   1.0/rho[tid+1*nx];
        b_j   =   1.0/rho[tid+1*nz*nx];
        b_x  =  2.0*b*b_i/(b+b_i);
        b_y  =  2.0*b*b_j/(b+b_j) ;
        b_z  =  2.0*b*b_k/(b+b_k) ;
        
        dTauxx_dx = Tauxx[tid+1] - Tauxx[tid];
        dTauzz_dz = Tauzz[tid+1*nx]- Tauzz[tid];
        dTauyy_dy = Tauyy[tid+1*nz*nx]- Tauyy[tid];
        
        dTauxy_dy = Tauxy[tid]  - Tauxy[tid-1*nz*nx];
        dTauxy_dx = Tauxy[tid]  - Tauxy[tid-1];
        dTauxz_dz = Tauxz[tid]  - Tauxz[tid-1*nx];
        dTauxz_dx = Tauxz[tid] - Tauxz[tid-1];
        dTauyz_dz = Tauyz[tid]- Tauyz[tid-1*nx];
        dTauyz_dy = Tauyz[tid]- Tauyz[tid-1*nz*nx];
       
        psidTauxx_dx[tid] = bx[x]*psidTauxx_dx[tid] + ax[x]*dTauxx_dx;
        psidTauxy_dx[tid] = bx[x]*psidTauxy_dx[tid] + ax[x]*dTauxy_dx;
        psidTauxz_dx[tid] = bx[x]*psidTauxz_dx[tid] + ax[x]*dTauxz_dx;
        
        psidTauyy_dy[tid] = by[y]*psidTauyy_dy[tid] + ay[y]*dTauyy_dy;
        psidTauxy_dy[tid] = by[y]*psidTauxy_dy[tid] + ay[y]*dTauxy_dy;
        psidTauyz_dy[tid] = by[y]*psidTauyz_dy[tid] + ay[y]*dTauyz_dy;
        
        psidTauzz_dz[tid] = bz[z]*psidTauzz_dz[tid] + az[z]*dTauzz_dz;
        psidTauxz_dz[tid] = bz[z]*psidTauxz_dz[tid] + az[z]*dTauxz_dz;
        psidTauyz_dz[tid] = bz[z]*psidTauyz_dz[tid] + az[z]*dTauyz_dz;
 
        Vx[tid] =Vx[tid]+ (dt/dx)*b_x*( (dTauxx_dx+psidTauxx_dx[tid]) + (dTauxy_dy+psidTauxy_dy[tid]) + (dTauxz_dz+psidTauxz_dz[tid]) );
        Vy[tid] =Vy[tid]+ (dt/dx)*b_y*( (dTauxy_dx+psidTauxy_dx[tid]) + (dTauyy_dy+psidTauyy_dy[tid]) + (dTauyz_dz+psidTauyz_dz[tid]) );
        Vz[tid] =Vz[tid]+ (dt/dx)*b_z*( (dTauxz_dx+psidTauxz_dx[tid]) + (dTauyz_dy+psidTauyz_dy[tid]) + (dTauzz_dz+psidTauzz_dz[tid]) );
    }
}

                             
__global__ void kernel_Gradients_Rho(float *bVx, float *bVy, float *bVz,
                                     float *fVx, float *fVy, float *fVz,  
                                     float *fVx_present, float *fVy_present, float *fVz_present,  
                                     float *vp, float *vs, float *rho,
                                     float *gkRho,
                                     int lenPML, int nt, int it, float dt,
                                     int nx, int ny, int nz){
                                    
    int x = idx;
    int y = idy;
    int z = idz;
    int tid = x + z * nx + y * nz * nx;
    
    if(x>=H_STENCIL && x<nx-H_STENCIL && z>=H_STENCIL && z<nz-H_STENCIL && y>=H_STENCIL && y<ny-H_STENCIL){

        gkRho[tid] =   -((bVx[tid]*(fVx_present[tid]-fVx[tid])/dt)+ (bVy[tid]*(fVy_present[tid]-fVy[tid])/dt) + (bVz[tid]*(fVz_present[tid]-fVz[tid])/dt));
        
    }
}
    

    
__global__ void kernel_Gradients_Lambda(float *bTauxx, float *bTauyy, float *bTauzz,
                                        float *fVx, float *fVy, float *fVz,   
                                        float *vp, float *vs, float *rho, float *lambda, float *mu,
                                        float *gkLambda, float dx,  float dy, float dz,
                                        int lenPML, int nt, int it, float dt,
                                        int nx, int ny, int nz){
                                    
    
    int x = idx;
    int y = idy;
    int z = idz;
    int tid = x + z * nx + y * nz * nx;
    
    
    float dVx_dx, dVy_dy, dVz_dz;
    float Mu, Lambda;

    if(x>=H_STENCIL && x<nx-H_STENCIL && z>=H_STENCIL && z<nz-H_STENCIL && y>=H_STENCIL && y<ny-H_STENCIL){
        Mu =     mu[tid];
        Lambda = lambda[tid];
        
        dVx_dx = 0.5*(fVx[tid] - fVx[tid-1])/dx +  0.5*(fVx[tid-1] - fVx[tid-2])/dx;
        dVy_dy = 0.5*(fVy[tid] - fVy[tid-1*nz*nx])/dy + 0.5*(fVy[tid-1*nz*nx] - fVy[tid-2*nz*nx])/dy;
        dVz_dz = 0.5*(fVz[tid] - fVz[tid-1*nx])/dz + 0.5*(fVz[tid-1*nx] - fVz[tid-2*nx])/dz;
        
        gkLambda[tid] =  (bTauxx[tid] + bTauyy[tid] + bTauzz[tid])*(dVx_dx + dVy_dy + dVz_dz)/( 3.0 *Lambda + 2*Mu);
    }
}

       
__global__ void kernel_Gradients_Mu(float *bTauxx,  float *bTauyy, float *bTauzz, 
                                    float *bTauxy,  float *bTauxz, float *bTauyz, 
                                    float *fVx, float *fVy, float *fVz,   
                                    float *vp, float *vs, float *rho, float *lambda, float *mu,
                                    float *gkMu,  float dx, float dy, float dz,
                                     int lenPML, int nt, int it, float dt,
                                    int nx, int ny, int nz){
                                    
    int x = idx;
    int y = idy;
    int z = idz;
    int tid = x + z * nx + y * nz * nx;
    
    
    float Lambda, Mu, Mu_i, Mu_j, Mu_k, Mu_ik, Mu_jk, Mu_ij,  Mu_xy, Mu_xz, Mu_yz;
    float dVx_dx, dVy_dy, dVz_dz;
    float dVx_dz, dVx_dy, dVy_dx, dVy_dz, dVz_dx, dVz_dy;
    if(x>=H_STENCIL && x<nx-H_STENCIL && z>=H_STENCIL && z<nz-H_STENCIL && y>=H_STENCIL && y<ny - H_STENCIL){
        Mu =     mu[tid];
        Lambda= lambda[tid];
        Mu_i  = mu[tid+1];
        Mu_k  = mu[tid+1*nx] ;
        Mu_j  = mu[tid+1*nz*nx];
        Mu_ik = mu[tid+1+1*nx] ;
        Mu_jk = mu[tid+1*nx+1*nz*nx];
        Mu_ij = mu[tid+1+1*nz*nx] ;

        Mu_xz=   1/(0.25/Mu+0.25/Mu_i+0.25/Mu_k+0.25/Mu_ik);
        Mu_xy=   1/(0.25/Mu+0.25/Mu_i+0.25/Mu_j+0.25/Mu_ij);
        Mu_yz=   1/(0.25/Mu+0.25/Mu_j+0.25/Mu_k+0.25/Mu_jk);
        
        dVx_dx = fVx[tid] - fVx[tid-1];
        dVy_dy = fVy[tid] - fVy[tid-1*nz*nx];
        dVz_dz = fVz[tid] - fVz[tid-1*nx];
        
        dVx_dz = fVx[tid+1*nx] - fVx[tid];
        dVx_dy = fVx[tid+1*nx*nz] - fVx[tid];
        dVy_dx = fVy[tid+1] - fVy[tid];
        dVy_dz = fVy[tid+1*nx] - fVy[tid];
        dVz_dx = fVz[tid+1]-fVz[tid];
        dVz_dy = fVz[tid+1*nz*nx]-fVz[tid];
        
        
        gkMu[tid] =   (1.0/( Mu * (3*Lambda + 2*Mu))) * (2*(Lambda + Mu)*bTauxx[tid]-Lambda*bTauyy[tid]-Lambda*bTauzz[tid]) * (dVx_dx)
                    + (1.0/( Mu * (3*Lambda + 2*Mu))) * (2*(Lambda + Mu)*bTauyy[tid]-Lambda*bTauxx[tid]-Lambda*bTauzz[tid]) * (dVy_dy)
                    + (1.0/( Mu * (3*Lambda + 2*Mu))) * (2*(Lambda + Mu)*bTauzz[tid]-Lambda*bTauxx[tid]-Lambda*bTauyy[tid]) * (dVz_dz)
                    + (1.0/ Mu_xy)*bTauxy[tid]*( dVy_dx + dVx_dy )
                    + (1.0/ Mu_xz)*bTauxz[tid]*( dVx_dz + dVz_dx )
                    + (1.0/ Mu_yz)*bTauyz[tid]*( dVz_dy + dVy_dz );
        
    }
}
    
__global__ void kernel_conv2D(float *in, int nx, int nz){
      int x = idx;
      int y = idy;
      int tid = idx + idy * nx;

      if(x>=1 && x<nx-1 && y>1 && y<nz-1){
          in[tid]=(
                       + 1.0/16.0 * in[ tid -1 +1*nx] + 1.0/8.0 * in[ tid-1 ] + 1.0/16.0 * in[ tid-1-1*nx ]
                       + 1.0/8.0 * in[ tid-1*nx]  + 1.0/4.0 * in[ tid] + 1.0/8.0 * in[ tid+1*nx ]
                       + 1.0/16.0 * in[ tid+1+1*nx ] + 1.0/8.0 * in[ tid+1 ] + 1.0/16.0 * in[ tid+1-1*nx ] );
     }
}
      

__global__ void kernel_add_adjoint_source(float *Vx, float *gatherVx,
                                          float *Vy, float *gatherVy, 
                                          float *Vz, float *gatherVz, 
                                          int *rec_pos_x,  int *rec_pos_y, int *rec_pos_z,  
                                          int nStations, int nt, int it, int nx, int nz){

    int x = idx;
    if(x<nStations){
        Vx[rec_pos_x[x] + nx*(rec_pos_z[x]) + nx*nz*(rec_pos_y[x])] = Vx[rec_pos_x[x] + nx*(rec_pos_z[x]) + nx*nz*(rec_pos_y[x])] + gatherVx[x+nStations*(nt-it-1)];
        Vy[rec_pos_x[x] + nx*(rec_pos_z[x]) + nx*nz*(rec_pos_y[x])] = Vy[rec_pos_x[x] + nx*(rec_pos_z[x]) + nx*nz*(rec_pos_y[x])] + gatherVy[x+nStations*(nt-it-1)];
        Vz[rec_pos_x[x] + nx*(rec_pos_z[x]) + nx*nz*(rec_pos_y[x])] = Vz[rec_pos_x[x] + nx*(rec_pos_z[x]) + nx*nz*(rec_pos_y[x])] + gatherVz[x+nStations*(nt-it-1)];
    }
}

__global__ void kernel_get_Station(float *Vx, float *gatherVx,
                                   float *Vy, float *gatherVy,  
                                   float *Vz, float *gatherVz, 
                                   int *rec_pos_x, int *rec_pos_y, int *rec_pos_z,  
                                   int nStations, int nt, int it, int nx, int nz){
    int x = idx;
    if(x<nStations){
        gatherVx[x+nStations*it] = Vx[rec_pos_x[x] + nx*(rec_pos_z[x]) + nx*nz*(rec_pos_y[x])];
        gatherVy[x+nStations*it] = Vy[rec_pos_x[x] + nx*(rec_pos_z[x]) + nx*nz*(rec_pos_y[x])];
        gatherVz[x+nStations*it] = Vz[rec_pos_x[x] + nx*(rec_pos_z[x]) + nx*nz*(rec_pos_y[x])];
    }
}

__global__ void kernel_Gradients_Mij(float *bTauxx, float *bTauyy, float *bTauzz, 
                                     float *bTauxy, float *bTauxz, float *bTauyz,
                                     float *ondicula, 
                                     float *vp, float *vs, float *rho, float *lambda, float *mu,
                                     float *M,
                                     int Sx,  int Sy, int Sz,
                                     int nt, int it,
                                     int nx, int nz){                             
    int x = idx;
    int tid = Sx + Sz * nx + Sy*nx*nz;
    float Lambda, Mu, Mu_i, Mu_j, Mu_k, Mu_ik, Mu_jk, Mu_ij,  Mu_xy, Mu_xz, Mu_yz;

        
    if(x<1){
        Mu =     mu[tid];
        Lambda = lambda[tid];
        Mu_i  =  mu[tid+1];
        Mu_k  =  mu[tid+1*nx] ;
        Mu_j  =  mu[tid+1*nz*nx];
        Mu_ik =  mu[tid+1+1*nx] ;
        Mu_jk =  mu[tid+1*nx+1*nz*nx];
        Mu_ij =  mu[tid+1+1*nz*nx] ;

        Mu_xz=   1/(0.25/Mu+0.25/Mu_i+0.25/Mu_k+0.25/Mu_ik);
        Mu_xy=   1/(0.25/Mu+0.25/Mu_i+0.25/Mu_j+0.25/Mu_ij);
        Mu_yz=   1/(0.25/Mu+0.25/Mu_j+0.25/Mu_k+0.25/Mu_jk);
        
        
        M[0] = ondicula[nt-it-1] * (1.0 / (2.0 * Mu * (3.0*Lambda + 2.0*Mu))) * (2.0*(Lambda + Mu) *  bTauxx[tid] - Lambda * bTauyy[tid] - Lambda * bTauzz[tid]);
        M[1] = ondicula[nt-it-1] * (1.0 / (2.0 * Mu * (3.0*Lambda + 2.0*Mu))) * (2.0*(Lambda + Mu) *  bTauyy[tid] - Lambda * bTauxx[tid] - Lambda * bTauzz[tid]);
        M[2] = ondicula[nt-it-1] * (1.0 / (2.0 * Mu * (3.0*Lambda + 2.0*Mu))) * (2.0*(Lambda + Mu) *  bTauzz[tid] - Lambda * bTauxx[tid] - Lambda * bTauyy[tid]);
        M[3] = ondicula[nt-it-1] * (1.0 / ( Mu_xy )) * ( bTauxy[tid] );
        M[4] = ondicula[nt-it-1] * (1.0 / ( Mu_xz )) * ( bTauxz[tid] );
        M[5] = ondicula[nt-it-1] * (1.0 / ( Mu_yz )) * ( bTauyz[tid] );
    }
}
    
    
        

    
__global__ void kernel_get_border(float *d_in,
	    				  float *d_front, float *d_back,
						  float *d_left, float *d_right,
						  float *d_top, float *d_bottom,
	   					  int nx, int ny, int nz, int iteration, int WCPML){

    int x = idx;
    int y = idy;
    int z = idz;
    int tid = x + z * nx + y * nz * nx;
    
    //------ LEFT ------//
    if(x >= WCPML && x < (WCPML+H_STENCIL) && z >= 0 && z < nz && y >= 0 && y < ny){
        d_left[x-WCPML +  H_STENCIL*z + y*H_STENCIL*nz + iteration*ny*nz*H_STENCIL] = d_in[tid] ;
    }
    
    //------ RIGHT ------//
    if(x >= (nx-WCPML-H_STENCIL) && x < (nx-WCPML) && z >= 0 && z < nz && y >= 0 && y < ny){
       d_right[x-(nx-WCPML-H_STENCIL) +  H_STENCIL*z + y*H_STENCIL*nz + iteration*ny*nz*H_STENCIL] =  d_in[tid];
    }
    
    //------ TOP ------//
    if( x >= 0 && x < nx && z >= H_STENCIL && z < (2*H_STENCIL) && y >= 0 && y < ny) {
		d_top[x +  (z-H_STENCIL)*nx + y*H_STENCIL*nx + iteration*ny*H_STENCIL*nx] = d_in[tid];
    }
    
    //------ BOTTOM ------//
	if( x >= 0 && x < nx && z >= (nz-WCPML-H_STENCIL) && z < (nz-WCPML) && y >= 0 && y < ny) {
  		d_bottom[x +  (z-(nz-WCPML-H_STENCIL))*nx + y*H_STENCIL*nx + iteration*ny*H_STENCIL*nx] = d_in[tid];
    }

    //------ FRONT ------//
	if( x >= 0 && x < nx && z >= 0 && z < nz && y >= WCPML && y < (WCPML+H_STENCIL) ) {
  		d_front[x +  z*nx + (y-WCPML)*nx*nz + iteration*H_STENCIL*nz*nx] = d_in[tid];
    }
    
    //------ BACK ------//
	if( x >= 0 && x < nx && z >= 0 && z < nz && y >= (ny-WCPML-H_STENCIL) && y < (ny-WCPML) ) {
  		d_back[x +  z*nx + (y-(ny-WCPML-H_STENCIL))*nx*nz + iteration*H_STENCIL*nz*nx] = d_in[tid];
    }
    
  }


__global__ void kernel_set_border(float *d_in,
  	    				  float *d_front, float *d_back,
  						  float *d_left, float *d_right,
  						  float *d_top, float *d_bottom,
  	   					  int nx, int ny, int nz, int iteration, int WCPML){

  	int x = idx;
    int y = idy;
    int z = idz;
    int tid = x + z * nx + y * nz * nx;
    
    //------ LEFT ------//
    if(x >= WCPML && x < (WCPML+H_STENCIL) && z >= 0 && z < nz && y >= 0 && y < ny){
        d_in[tid] = d_left[x-WCPML +  H_STENCIL*z + y*H_STENCIL*nz + iteration*ny*nz*H_STENCIL] ;
    }
    
    //------ RIGHT ------//
    if(x >= (nx-WCPML-H_STENCIL) && x < (nx-WCPML) && z >= 0 && z < nz && y >= 0 && y < ny){
        d_in[tid] = d_right[x-(nx-WCPML-H_STENCIL) +  H_STENCIL*z + y*H_STENCIL*nz + iteration*ny*nz*H_STENCIL];
    }
    
    //------ TOP ------//
    if( x >= 0 && x < nx && z >= H_STENCIL && z < (2*H_STENCIL) && y >= 0 && y < ny) {
		d_in[tid] = d_top[x +  (z-H_STENCIL)*nx + y*H_STENCIL*nx + iteration*ny*H_STENCIL*nx];
    }
    
    //------ BOTTOM ------//
	if( x >= 0 && x < nx && z >= (nz-WCPML-H_STENCIL) && z < (nz-WCPML) && y >= 0 && y < ny) {
  		d_in[tid] =  d_bottom[x +  (z-(nz-WCPML-H_STENCIL))*nx + y*H_STENCIL*nx + iteration*ny*H_STENCIL*nx];
    }

    //------ FRONT ------//
	if( x >= 0 && x < nx && z >= 0 && z < nz && y >= WCPML && y < (WCPML+H_STENCIL) ) {
  		d_in[tid] = d_front[x +  z*nx + (y-WCPML)*nx*nz + iteration*H_STENCIL*nz*nx];
    }
    
    //------ BACK ------//
	if( x >= 0 && x < nx && z >= 0 && z < nz && y >= (ny-WCPML-H_STENCIL) && y < (ny-WCPML) ) {
  		d_in[tid] = d_back[x +  z*nx + (y-(ny-WCPML-H_STENCIL))*nx*nz + iteration*H_STENCIL*nz*nx];
    }
}
    
    

__global__ void kernel_setZero_CPMLzone(float *d_in,
  	    				    			int nx, int ny, int nz, int WCPML){

  	int x = idx;
    int y = idy;
    int z = idz;
    int tid = x + z * nx + y * nz * nx;
    
    //------ LEFT ------//
    if(x >= 0 && x < WCPML  && z >= 0 && z < nz && y >= 0 && y < ny){
        d_in[tid] = 0.0;
    }
    
    //------ RIGHT ------//
    if(x >= nx-WCPML && x < nx && z >= 0 && z < nz && y >= 0 && y < ny){
        d_in[tid] = 0.0;
    }
    
    
    //------ BOTTOM ------//
	if( x >= 0 && x < nx && z >= (nz-WCPML) && z < nz && y >= 0 && y < ny) {
  		d_in[tid] =  0.0;
    }

    //------ FRONT ------//
	if( x >= 0 && x < nx && z >= 0 && z < nz && y >= 0  && y < WCPML ) {
  		d_in[tid] = 0.0;
    }
    
    //------ BACK ------//
	if( x >= 0 && x < nx && z >= 0 && z < nz && y >= (ny-WCPML) && y < ny ) {
  		d_in[tid] = 0.0;
    }
}
    
__global__ void kernel_DTD_copy(float *d_odata, float *d_idata, int nx, int ny, int nz){

    int x = idx;
    int y = idy;
    int z = idz;
    int tid = x + z * nx + y * nz * nx;

    if(x<nx && y<ny && z<nz){
        d_odata[tid] = d_idata[tid];
    }
}


""")

#----------------------------------------------------------------------------------------------------------------------#
''' Kernels definitions '''
kernel_Tau = ker.get_function("kernel_Tau")
kernel_Vx_Vy_Vz = ker.get_function("kernel_Vx_Vy_Vz")
kernel_conv2D = ker.get_function("kernel_conv2D")
kernel_add_adjoint_source = ker.get_function("kernel_add_adjoint_source")
kernel_get_Station = ker.get_function("kernel_get_Station")
kernel_Gradients_Mij = ker.get_function("kernel_Gradients_Mij")
kernel_Gradients_Rho = ker.get_function("kernel_Gradients_Rho")
kernel_Gradients_Lambda = ker.get_function("kernel_Gradients_Lambda")
kernel_Gradients_Mu = ker.get_function("kernel_Gradients_Mu")
kernel_get_border = ker.get_function("kernel_get_border")
kernel_set_border = ker.get_function("kernel_set_border")
kernel_setZero_CPMLzone = ker.get_function("kernel_setZero_CPMLzone")
kernel_DTD_copy = ker.get_function("kernel_DTD_copy")


#----------------------------------------------------------------------------------------------------------------------#
''' Class definitions '''

class Station():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"x:{self.x} y:{self.y}, z:{self.z}"
    
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getZ(self):
        return self.z

    def setX(self, x):
        self.x = x
    def setY(self, y):
        self.y = y
    def setZ(self, z):
        self.z = z


class Earthquake():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.stations = []
        self.mag = None

    def __str__(self):
        return f"x:{self.x} y:{self.y}, z:{self.z}"

    def addStation(self, newStation):
        self.stations.append(newStation)

    def printStationsSismo(self):
        print(f"{30 * '-'}")
        print(f"{'# estacion':<10}| {'Coordenadas:':<20}")
        print(f"{30 * '-'}")

        for i in range(len(self.stations)):
            print(f"{i + 1:< 10}| {self.stations[i].__str__():<20}")
            print(f"{30 * '-'}")

    def getLenStations(self):
        return len(self.stations)

    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getZ(self):
        return self.z
    def getMag(self):
        return self.mag

    def setX(self, x_in):
        self.x = x_in
    def setY(self, y_in):
        self.y = y_in
    def setZ(self, z_in):
        self.z = z_in
    def setMag(self, mag_in):
        self.mag = mag_in

class Geometry():

    def __init__(self):
        self.sismos = []

    def addSismo(self, newSismo):
        self.sismos.append(newSismo)

    def printSismos(self):
        print(f"{30*'-'}")
        print(f"{'# sismo':<10}| {'Coordenadas:':<20}")
        print(f"{30 * '-'}")

        for i in range(len(self.sismos)):
            print(f"{i+1:< 10}| {self.sismos[i].__str__():<20}")
            print(f"{30 * '-'}")

    def getLenSismos(self):
        return len(self.sismos)

    def getStationsEarthquake(self,  idxSismo):
        sta_pos_x = np.empty([])
        sta_pos_y = np.empty([])
        sta_pos_z = np.empty([])

        sta_pos_x = np.delete(sta_pos_x, 0)
        sta_pos_y = np.delete(sta_pos_y, 0)
        sta_pos_z = np.delete(sta_pos_z, 0)


        for j in range(len(self.sismos[idxSismo].stations)):
            sta_pos_x = np.append(sta_pos_x, self.sismos[idxSismo].stations[j].getX())
            sta_pos_y = np.append(sta_pos_y, self.sismos[idxSismo].stations[j].getY())
            sta_pos_z = np.append(sta_pos_z, self.sismos[idxSismo].stations[j].getZ())

        return sta_pos_x, sta_pos_y, sta_pos_z

    def printStations(self,  idxSismo= 0):
        self.sismos[idxSismo].printStationsSismo()

    def graficar(self, idxSismo=-1):

        sta_pos_x = np.empty([])
        sta_pos_y = np.empty([])
        sta_pos_z = np.empty([])

        sismo_pos_x = np.empty([])
        sismo_pos_y = np.empty([])
        sismo_pos_z = np.empty([])

        sta_pos_x = np.delete(sta_pos_x, 0)
        sta_pos_y = np.delete(sta_pos_y, 0)
        sta_pos_z = np.delete(sta_pos_z, 0)

        sismo_pos_x = np.delete(sismo_pos_x, 0)
        sismo_pos_y = np.delete(sismo_pos_y, 0)
        sismo_pos_z = np.delete(sismo_pos_z, 0)

        if idxSismo != -1:
            sismo_pos_x = self.sismos[idxSismo].getX()
            sismo_pos_y = self.sismos[idxSismo].getY()
            sismo_pos_z = self.sismos[idxSismo].getZ()

            for j in range(len(self.sismos[idxSismo].stations)):
                sta_pos_x = np.append(sta_pos_x, self.sismos[idxSismo].stations[j].getX())
                sta_pos_y = np.append(sta_pos_y, self.sismos[idxSismo].stations[j].getY())
                sta_pos_z = np.append(sta_pos_z, self.sismos[idxSismo].stations[j].getZ())

        else:
            for i in range(len(self.sismos)):
                sismo_pos_x = np.append(sismo_pos_x, self.sismos[i].getX())
                sismo_pos_y = np.append(sismo_pos_y, self.sismos[i].getY())
                sismo_pos_z = np.append(sismo_pos_z, self.sismos[i].getZ())

                for j in range(len(self.sismos[i].stations)):
                    sta_pos_x = np.append(sta_pos_x, self.sismos[i].stations[j].getX())
                    sta_pos_y = np.append(sta_pos_y, self.sismos[i].stations[j].getY())
                    sta_pos_z = np.append(sta_pos_z, self.sismos[i].stations[j].getZ())


        plt.figure()
        figax = plt.axes(projection='3d')
        figax.scatter3D(sta_pos_x, sta_pos_y, -sta_pos_z, cmap='Greens', label='Estaciones')
        figax.scatter3D(sismo_pos_x, sismo_pos_y, -sismo_pos_z, cmap='Greens', label='Sismos')
        figax.set_xlabel('X', size=18)
        figax.set_ylabel('Y', size=18)
        figax.set_zlabel('Z', size=18)
        figax.legend()
        figax.set(xlim=(0, nx), ylim=(0, ny), zlim=(-nz, 0))
#        plt.show()
        plt.show(block=False)
        plt.pause(5.0)
        plt.close()

    def save(self):

        sta_pos_x = np.empty([])
        sta_pos_y = np.empty([])
        sta_pos_z = np.empty([])

        sismo_pos_x = np.empty([])
        sismo_pos_y = np.empty([])
        sismo_pos_z = np.empty([])

        sta_pos_x = np.delete(sta_pos_x, 0)
        sta_pos_y = np.delete(sta_pos_y, 0)
        sta_pos_z = np.delete(sta_pos_z, 0)

        sismo_pos_x = np.delete(sismo_pos_x, 0)
        sismo_pos_y = np.delete(sismo_pos_y, 0)
        sismo_pos_z = np.delete(sismo_pos_z, 0)

        for i in range(len(self.sismos)):
            sismo_pos_x = np.append(sismo_pos_x, self.sismos[i].getX())
            sismo_pos_y = np.append(sismo_pos_y, self.sismos[i].getY())
            sismo_pos_z = np.append(sismo_pos_z, self.sismos[i].getZ())

            for j in range(len(self.sismos[i].stations)):
                sta_pos_x = np.append(sta_pos_x, self.sismos[i].stations[j].getX())
                sta_pos_y = np.append(sta_pos_y, self.sismos[i].stations[j].getY())
                sta_pos_z = np.append(sta_pos_z, self.sismos[i].stations[j].getZ())

        save('Results/Actual/Geometry/sismos_pos_x.npy', sismo_pos_x)
        save('Results/Actual/Geometry/sismos_pos_y.npy', sismo_pos_y)
        save('Results/Actual/Geometry/sismos_pos_z.npy', sismo_pos_z)
        save('Results/Actual/Geometry/sta_pos_x.npy', sta_pos_x)
        save('Results/Actual/Geometry/sta_pos_y.npy', sta_pos_y)
        save('Results/Actual/Geometry/sta_pos_z.npy', sta_pos_z)

class Borders():
    def __init__(self, ny, nz, nx, nt):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nt = nt
        self.left = gpuarray.to_gpu(np.zeros((nt, ny, nz, H_STENCIL)).astype(np.float32))
        self.right = gpuarray.to_gpu(np.zeros((nt, ny, nz, H_STENCIL)).astype(np.float32))
        self.top = gpuarray.to_gpu(np.zeros((nt, ny, H_STENCIL, nx)).astype(np.float32))
        self.bottom = gpuarray.to_gpu(np.zeros((nt, ny, H_STENCIL, nx)).astype(np.float32))
        self.front = gpuarray.to_gpu(np.zeros((nt, H_STENCIL, nz, nx)).astype(np.float32))
        self.back = gpuarray.to_gpu(np.zeros((nt, H_STENCIL, nz, nx)).astype(np.float32))
        
    def setZero(self):
        self.left = gpuarray.to_gpu(np.zeros((self.nt, ny, nz, H_STENCIL)).astype(np.float32))
        self.right = gpuarray.to_gpu(np.zeros((self.nt, ny, nz, H_STENCIL)).astype(np.float32))
        self.top = gpuarray.to_gpu(np.zeros((self.nt, ny, H_STENCIL, nx)).astype(np.float32))
        self.bottom = gpuarray.to_gpu(np.zeros((self.nt, ny, H_STENCIL, nx)).astype(np.float32))
        self.front = gpuarray.to_gpu(np.zeros((self.nt, H_STENCIL, nz, nx)).astype(np.float32))
        self.back = gpuarray.to_gpu(np.zeros((self.nt, H_STENCIL, nz, nx)).astype(np.float32))

class Fields():
    def __init__(self, ny, nz, nx, nt):
        # ============================= GPU Memory ============================== #
        # ---> Vx, Vy, Vz
        self.Vx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Vy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Vz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> Tauxx, Tauyy, Tauzz, Tauxy, Tauxz, Tauyz
        self.Tauxx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Tauyy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Tauzz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Tauxy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Tauxz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Tauyz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        # ---> Vx_present, Vy_present, Vz_present
        self.Vx_present = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Vy_present = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Vz_present = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        # ============================= GPU Memory ============================== #
        # ---> psidVx_dz, psidVy_dz, psidVz_dz
        self.psidVx_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVy_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVz_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> psidVy_dx, psidVz_dx, psidVx_dx
        self.psidVy_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVz_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVx_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> psidVx_dy, psidVz_dy, psidVy_dy
        self.psidVx_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVz_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVy_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ============================= GPU Memory ============================== #
        # ---> psidTauzz_dz, psidTauyz_dz, psidTauxz_dz
        self.psidTauzz_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauyz_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauxz_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> psidTauxx_dx, psidTauxy_dx, psidTauxz_dx
        self.psidTauxx_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauxy_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauxz_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> psidTauyy_dy, psidTauxy_dy, psidTauyz_dy
        self.psidTauyy_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauxy_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauyz_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ============================= CPU Memory ============================== #
        # ---> h_psidVx_dz, h_psidVy_dz, h_psidVz_dz
        self.h_psidVx_dz = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidVy_dz = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidVz_dz = np.zeros((ny, nz, nx)).astype(np.float32)

        # ---> h_psidVy_dx, h_psidVz_dx, h_psidVx_dx
        self.h_psidVy_dx = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidVz_dx = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidVx_dx = np.zeros((ny, nz, nx)).astype(np.float32)

        # ---> h_psidVx_dy, h_psidVz_dy, h_psidVy_dy
        self.h_psidVx_dy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidVz_dy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidVy_dy = np.zeros((ny, nz, nx)).astype(np.float32)

        # ============================= CPU Memory ============================== #
        # ---> psidTauzz_dz, psidTauyz_dz, psidTauxz_dz
        self.h_psidTauzz_dz = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidTauyz_dz = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidTauxz_dz = np.zeros((ny, nz, nx)).astype(np.float32)

        # ---> psidTauxx_dx, psidTauxy_dx, psidTauxz_dx
        self.h_psidTauxx_dx = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidTauxy_dx = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidTauxz_dx = np.zeros((ny, nz, nx)).astype(np.float32)

        # ---> psidTauyy_dy, psidTauxy_dy, psidTauyz_dy
        self.h_psidTauyy_dy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidTauxy_dy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidTauyz_dy = np.zeros((ny, nz, nx)).astype(np.float32)

        # ============================= CPU Memory ============================== #

        # ---> Vx, Vy, Vz
        self.h_Vx = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Vy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Vz = np.zeros((ny, nz, nx)).astype(np.float32)

        # ---> Tauxx, Tauyy, Tauzz, Tauxy, Tauxz, Tauyz
        self.h_Tauxx = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Tauyy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Tauzz = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Tauxy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Tauxz = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Tauyz = np.zeros((ny, nz, nx)).astype(np.float32)
        
        # ============================= Borders ================================= #
        self.VxBorders = Borders(ny, nz, nx, nt)
        self.VyBorders = Borders(ny, nz, nx, nt)
        self.VzBorders = Borders(ny, nz, nx, nt)
       
#        self.TauxxBorders = Borders(ny, nz, nx, nt)
#        self.TauyyBorders = Borders(ny, nz, nx, nt)
#        self.TauzzBorders = Borders(ny, nz, nx, nt)
#        self.TauxyBorders = Borders(ny, nz, nx, nt)
#        self.TauxzBorders = Borders(ny, nz, nx, nt)
#        self.TauyzBorders = Borders(ny, nz, nx, nt)

    def setZero(self):
        # ---> Vx, Vy, Vz
        self.Vx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Vy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Vz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> Tauxx, Tauyy, Tauzz, Tauxy, Tauxz, Tauyz
        self.Tauxx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Tauyy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Tauzz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Tauxy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Tauxz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Tauyz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        
        # ---> Vx_present, Vy_present, Vz_present
        self.Vx_present = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Vy_present = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.Vz_present = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        # ============================= GPU Memory ============================== #
        # ---> psidVx_dz, psidVy_dz, psidVz_dz
        self.psidVx_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVy_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVz_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> psidVy_dx, psidVz_dx, psidVx_dx
        self.psidVy_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVz_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVx_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> psidVx_dy, psidVz_dy, psidVy_dy
        self.psidVx_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVz_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVy_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ============================= GPU Memory ============================== #
        # ---> psidTauzz_dz, psidTauyz_dz, psidTauxz_dz
        self.psidTauzz_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauyz_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauxz_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> psidTauxx_dx, psidTauxy_dx, psidTauxz_dx
        self.psidTauxx_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauxy_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauxz_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> psidTauyy_dy, psidTauxy_dy, psidTauyz_dy
        self.psidTauyy_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauxy_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauyz_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ============================= CPU Memory ============================== #
        # ---> h_psidVx_dz, h_psidVy_dz, h_psidVz_dz
        self.h_psidVx_dz = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidVy_dz = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidVz_dz = np.zeros((ny, nz, nx)).astype(np.float32)

        # ---> h_psidVy_dx, h_psidVz_dx, h_psidVx_dx
        self.h_psidVy_dx = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidVz_dx = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidVx_dx = np.zeros((ny, nz, nx)).astype(np.float32)

        # ---> h_psidVx_dy, h_psidVz_dy, h_psidVy_dy
        self.h_psidVx_dy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidVz_dy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidVy_dy = np.zeros((ny, nz, nx)).astype(np.float32)

        # ============================= CPU Memory ============================== #
        # ---> psidTauzz_dz, psidTauyz_dz, psidTauxz_dz
        self.h_psidTauzz_dz = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidTauyz_dz = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidTauxz_dz = np.zeros((ny, nz, nx)).astype(np.float32)

        # ---> psidTauxx_dx, psidTauxy_dx, psidTauxz_dx
        self.h_psidTauxx_dx = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidTauxy_dx = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidTauxz_dx = np.zeros((ny, nz, nx)).astype(np.float32)

        # ---> psidTauyy_dy, psidTauxy_dy, psidTauyz_dy
        self.h_psidTauyy_dy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidTauxy_dy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_psidTauyz_dy = np.zeros((ny, nz, nx)).astype(np.float32)

        # ============================= CPU Memory ============================== #
        # ---> Vx, Vy, Vz
        self.h_Vx = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Vy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Vz = np.zeros((ny, nz, nx)).astype(np.float32)

        # ---> Tauxx, Tauyy, Tauzz, Tauxy, Tauxz, Tauyz
        self.h_Tauxx = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Tauyy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Tauzz = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Tauxy = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Tauxz = np.zeros((ny, nz, nx)).astype(np.float32)
        self.h_Tauyz = np.zeros((ny, nz, nx)).astype(np.float32)
            
        # ============================= Borders ================================= #
        self.VxBorders.setZero()
        self.VyBorders.setZero()
        self.VzBorders.setZero()
        
     
#        self.TauxxBorders.setZero()
#        self.TauyyBorders.setZero()
#        self.TauzzBorders.setZero()
#        self.TauxyBorders.setZero()
#        self.TauxzBorders.setZero()
#        self.TauyzBorders.setZero()


    def setZeroPsi(self):
        # ============================= GPU Memory ============================== #
        # ---> psidVx_dz, psidVy_dz, psidVz_dz
        self.psidVx_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVy_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVz_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> psidVy_dx, psidVz_dx, psidVx_dx
        self.psidVy_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVz_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVx_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> psidVx_dy, psidVz_dy, psidVy_dy
        self.psidVx_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVz_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidVy_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ============================= GPU Memory ============================== #
        # ---> psidTauzz_dz, psidTauyz_dz, psidTauxz_dz
        self.psidTauzz_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauyz_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauxz_dz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> psidTauxx_dx, psidTauxy_dx, psidTauxz_dx
        self.psidTauxx_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauxy_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauxz_dx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        # ---> psidTauyy_dy, psidTauxy_dy, psidTauyz_dy
        self.psidTauyy_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauxy_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.psidTauyz_dy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        
    def setZeroCPMLzone(self, ny, nz, nx, WCPML):
        
        
        kernel_setZero_CPMLzone(self.Vx, 
                          np.int32(nx), np.int32(ny), np.int32(nz), np.int32(WCPML),
                          grid=(int(nx / threadBlockX) + 1, 
                                int(ny / threadBlockY) + 1,
                                int(nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
                          
        kernel_setZero_CPMLzone(self.Vy, 
                          np.int32(nx), np.int32(ny), np.int32(nz), np.int32(WCPML),
                          grid=(int(nx / threadBlockX) + 1, 
                                int(ny / threadBlockY) + 1,
                                int(nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
        
        kernel_setZero_CPMLzone(self.Vz, 
                          np.int32(nx), np.int32(ny), np.int32(nz), np.int32(WCPML),
                          grid=(int(nx / threadBlockX) + 1, 
                                int(ny / threadBlockY) + 1,
                                int(nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))

        kernel_setZero_CPMLzone(self.Tauxx, 
                          np.int32(nx), np.int32(ny), np.int32(nz), np.int32(WCPML),
                          grid=(int(nx / threadBlockX) + 1, 
                                int(ny / threadBlockY) + 1,
                                int(nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
        
        kernel_setZero_CPMLzone(self.Tauyy, 
                          np.int32(nx), np.int32(ny), np.int32(nz), np.int32(WCPML),
                          grid=(int(nx / threadBlockX) + 1, 
                                int(ny / threadBlockY) + 1,
                                int(nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))

        kernel_setZero_CPMLzone(self.Tauzz, 
                          np.int32(nx), np.int32(ny), np.int32(nz), np.int32(WCPML),
                          grid=(int(nx / threadBlockX) + 1, 
                                int(ny / threadBlockY) + 1,
                                int(nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))

        kernel_setZero_CPMLzone(self.Tauxy, 
                          np.int32(nx), np.int32(ny), np.int32(nz), np.int32(WCPML),
                          grid=(int(nx / threadBlockX) + 1, 
                                int(ny / threadBlockY) + 1,
                                int(nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))

        kernel_setZero_CPMLzone(self.Tauxz, 
                          np.int32(nx), np.int32(ny), np.int32(nz), np.int32(WCPML),
                          grid=(int(nx / threadBlockX) + 1, 
                                int(ny / threadBlockY) + 1,
                                int(nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))

        kernel_setZero_CPMLzone(self.Tauyz, 
                          np.int32(nx), np.int32(ny), np.int32(nz), np.int32(WCPML),
                          grid=(int(nx / threadBlockX) + 1, 
                                int(ny / threadBlockY) + 1,
                                int(nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))                          
        # ---> Tauxx, Tauyy, Tauzz, Tauxy, Tauxz, Tauyz
#        self.Tauxx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
#        self.Tauyy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
#        self.Tauzz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
#        self.Tauxy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
#        self.Tauxz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
#        self.Tauyz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

        
        

    def hostMemAllocFields(self, ny, nz, nx, nt):
        pass
#        self.h_fVx = np.zeros((ny, nz, nx, nt)).astype(np.float32)
#        self.h_fVy = np.zeros((ny, nz, nx, nt)).astype(np.float32)
#        self.h_fVz = np.zeros((ny, nz, nx, nt)).astype(np.float32)
#     
#        self.d_fVx = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
#        self.d_fVy = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
#        self.d_fVz = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
    
    @staticmethod
    def graficar(self, fields, name,  corte_X, corte_Y, corte_Z):
        
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 2, 1)
        plt.imshow(fields.get()[:, :, corte_X], aspect="auto")
        plt.title(f'${name}~ corte~ X$', size=14)
        plt.ylabel('Y (Km)')
        plt.xlabel('Z (Km)')
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(fields.get()[:, corte_Z, :], aspect="auto")
        plt.title(f'${name}~ corte~ Z$', size=14)
        plt.ylabel('Y (Km)')
        plt.xlabel('X (Km)')
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.imshow(fields.get()[corte_Y, :, :], aspect="auto")
        plt.title(f'${name}~ corte~ Y$', size=14)
        plt.ylabel('Z (Km)')
        plt.xlabel('X (Km)')
        plt.colorbar()
           


def saveGif(cont):
    filenames = []

    with imageio.get_writer('gif/Propagacion3DFTDT.gif', mode='I') as writer:
        for i in range(1, cont + 1):
            filenames.append(f'Imagenes/Vx_{i}.png')

        #        print(filenames)

        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


class Gather():
    def __init__(self, nSamples, nStations):
        self.nSamples = nSamples
        self.nStations = nStations
        self.Vx = np.zeros((nSamples, nStations)).astype(np.float32)
        self.Vy = np.zeros((nSamples, nStations)).astype(np.float32)
        self.Vz = np.zeros((nSamples, nStations)).astype(np.float32)
        self.d_Vx = gpuarray.to_gpu(np.zeros((nSamples, nStations)).astype(np.float32))
        self.d_Vy = gpuarray.to_gpu(np.zeros((nSamples, nStations)).astype(np.float32))
        self.d_Vz = gpuarray.to_gpu(np.zeros((nSamples, nStations)).astype(np.float32))

    def setZero(self):
        self.Vx = np.zeros((self.nSamples, self.nStations)).astype(np.float32)
        self.Vy = np.zeros((self.nSamples, self.nStations)).astype(np.float32)
        self.Vz = np.zeros((self.nSamples, self.nStations)).astype(np.float32)

        self.d_Vx = gpuarray.to_gpu(np.zeros((self.nSamples, self.nStations)).astype(np.float32))
        self.d_Vy = gpuarray.to_gpu(np.zeros((self.nSamples, self.nStations)).astype(np.float32))
        self.d_Vz = gpuarray.to_gpu(np.zeros((self.nSamples, self.nStations)).astype(np.float32))

    def hostToDevice(self):
        self.d_Vx = gpuarray.to_gpu(self.Vx).astype(np.float32)
        self.d_Vy = gpuarray.to_gpu(self.Vy).astype(np.float32)
        self.d_Vz = gpuarray.to_gpu(self.Vz).astype(np.float32)

    def graficar(self, strTitle="Gather"):
        plt.subplot(1, 3, 1)
        Vclip = [np.amax(self.Vx), np.amin(self.Vx)]
        plt.imshow(self.Vx, aspect="auto", vmax=1e-2 * Vclip[0], vmin=1e-2 * Vclip[1])
        plt.title(strTitle+' Vx')

        plt.subplot(1, 3, 2)
        Vclip = [np.amax(self.Vy), np.amin(self.Vy)]
        plt.imshow(self.Vy, aspect="auto", vmax=1e-2 * Vclip[0], vmin=1e-2 * Vclip[1])
        plt.title(strTitle+' Vy')

        plt.subplot(1, 3, 3)
        Vclip = [np.amax(self.Vz), np.amin(self.Vz)]
        plt.imshow(self.Vz, aspect="auto", vmax=1e-2 * Vclip[0], vmin=1e-2 * Vclip[1])
        plt.title(strTitle+' Vz')

        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

class Model():
    def __init__(self, nz, ny, nx, dz, dy, dx):
        self.Vp = np.zeros((ny, nz, nx)).astype(np.float32)
        self.Vs = np.zeros((ny, nz, nx)).astype(np.float32)
        self.Rho = np.zeros((ny, nz, nx)).astype(np.float32)
        self.Mu = np.zeros((ny, nz, nx)).astype(np.float32)
        self.Lambda = np.zeros((ny, nz, nx)).astype(np.float32)
        self.d_Vp = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.d_Vs = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.d_Rho = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.d_Mu = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.d_Lambda = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dz = dz
        self.dy = dy
        self.dx = dx

    def compute_Lambda_Mu(self):
        self.Mu = self.Vs.copy() * self.Vs.copy() * self.Rho.copy()
        self.Lambda = self.Vp.copy() * self.Vp.copy() * self.Rho.copy() - 2.0 * self.Mu.copy()

    def compute_Vp_Vs(self):
        self.Vs = np.sqrt(self.Mu.copy()/self.Rho.copy())
        self.Vp = np.sqrt((self.Lambda.copy()+2.0*self.Mu.copy()) / self.Rho.copy())
    
    def compute_Rho(self):
        self.Rho = (1.6612 * ( self.Vp.copy() / 1e3) - 0.4712 * ((self.Vp.copy() / 1e3) ** 2) + 0.0671 * ((self.Vp.copy() / 1e3) ** 3) - 0.0043 * (
                 (self.Vp.copy() / 1e3) ** 4) + 0.000106 * ((self.Vp.copy() / 1e3) ** 5))*1e3

    def MaxMinVp(self):
        self.VmaxVp = np.amax(self.Vp)
        self.VminVp = np.amin(self.Vp)

    def hostToDevice(self):
        self.d_Vp = gpuarray.to_gpu(self.Vp.copy()).astype(np.float32)
        self.d_Vs = gpuarray.to_gpu(self.Vs.copy()).astype(np.float32)
        self.d_Rho = gpuarray.to_gpu(self.Rho.copy()).astype(np.float32)
        self.d_Mu = gpuarray.to_gpu(self.Mu.copy()).astype(np.float32)
        self.d_Lambda = gpuarray.to_gpu(self.Lambda.copy()).astype(np.float32)
    
    @staticmethod
    def graficar(model, name, corte_x, corte_y, corte_z):
       
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(model[:, :, corte_x], vmin=np.min(model), vmax=np.max(model))
        plt.title(f'${name}~ corte~ X$', size=14)
        plt.ylabel('Y (Km)')
        plt.xlabel('Z (Km)')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.imshow(model[:, corte_z, :], vmin=np.min(model), vmax=np.max(model))
        plt.title(f'${name}~ corte~ Z$', size=14)
        plt.ylabel('Y (Km)')
        plt.xlabel('X (Km)')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.imshow(model[corte_y, :, :], vmin=np.min(model), vmax=np.max(model))
        plt.title(f'${name} ~ corte~ Y$', size=14)
        plt.ylabel('Z (Km)')
        plt.xlabel('X (Km)')
        plt.colorbar()
#        plt.show()
        plt.show(block=False)
        plt.pause(3.0)
        plt.close()


class Wavelet():
    def __init__(self, fq, tprop, cn):
        self.tprop = tprop
        self.fq = fq
#        self.dt = 0.8 / (cn.VmaxVp * 1.1667 * sqrt(1.0 / (cn.dx * cn.dx) + 1.0 / (cn.dy * cn.dy) + 1.0 / (cn.dz * cn.dz)))
        self.dt = 0.08551979923718353
        self.nSamples = int(self.tprop / self.dt)
        self.waveform = np.zeros((self.nSamples, 1)).astype(np.float32)
        self.t = np.zeros((self.nSamples, 1)).astype(np.float32)
        self.d_waveform = gpuarray.to_gpu(np.zeros((self.nSamples, 1)).astype(np.float32))
        
        if rank == 0: print(f"dt:{self.dt} nSamples:{self.nSamples}")

    def hostToDevice(self):
        self.d_waveform = gpuarray.to_gpu(self.waveform).astype(np.float32)

    def ricker(self):
        t0 = (np.sqrt(2) / self.fq) / self.dt

        for t in range(self.nSamples):
            self.t[t] = t*self.dt
            self.waveform[t] = (1 - 2 * (pi * self.fq * ((t - t0) * self.dt)) ** 2) * exp(-(pi * fq * ((t - t0) * self.dt)) ** 2)

        self.hostToDevice()

    def gaussian(self):
        t0 = (1.0 / self.fq) / self.dt

        for t in range(self.nSamples):
            a = 2* (np.pi)**2 * (self.fq)**2
            self.t[t] = t*self.dt
            self.waveform[t] = exp(-a*( (t - t0) * self.dt) ** 2)
        self.hostToDevice()

    def graficar(self):
        plt.figure()
        plt.plot(self.t, self.waveform)
        plt.xlabel('Time [s]', size=18)
        plt.ylabel('Amplitude', size=18)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

    def __str__(self):
        return f"---- Parametros Ondicula ---- \n" \
               f"Samples: {str(self.nSamples)}\n" \
               f"Sample time: {self.dt:e} [s] \n" \
               f"Simulation time: {self.tprop} [s]"

class CPML():
    def __init__(self, lenPML, ondicula, cn):
        self.lenPML = lenPML 
        self.ax, self.bx = self.CPML_i(ondicula.fq, ondicula.dt, cn.dx, lenPML, cn.VmaxVp, cn.nx)
        self.ay, self.by = self.CPML_i(ondicula.fq, ondicula.dt, cn.dy, lenPML, cn.VmaxVp, cn.ny)
        self.az, self.bz = self.CPML_i(ondicula.fq, ondicula.dt, cn.dz, lenPML, cn.VmaxVp, cn.nz)

        self.az[:lenPML] = 0.0
        self.bz[:lenPML] = 0.0

        self.d_ax = gpuarray.to_gpu(self.ax).astype(np.float32)
        self.d_bx = gpuarray.to_gpu(self.bx).astype(np.float32)
        self.d_ay = gpuarray.to_gpu(self.ay).astype(np.float32)
        self.d_by = gpuarray.to_gpu(self.by).astype(np.float32)
        self.d_az = gpuarray.to_gpu(self.az).astype(np.float32)
        self.d_bz = gpuarray.to_gpu(self.bz).astype(np.float32)


    def graficar(self):
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.plot(self.d_ax.get())
        plt.title('ax')
        plt.subplot(2, 3, 4)
        plt.plot(self.d_bx.get())
        plt.title('bx')
        plt.subplot(2, 3, 2)
        plt.plot(self.d_ay.get())
        plt.title('ay')
        plt.subplot(2, 3, 5)
        plt.plot(self.d_by.get())
        plt.title('by')
        plt.subplot(2, 3, 3)
        plt.plot(self.d_az.get())
        plt.title('az')
        plt.subplot(2, 3, 6)
        plt.plot(self.d_bz.get())
        plt.title('bz')
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

    def CPML_i(self, frequency, dt, di, WCPML, velMax, Ni):
        R = 1e-6
        Li = WCPML * di
        d0 = -3 * np.log(R) / (2.0 * Li)
        Fi = 0.0
        d_i = 0.0
        Alphai = 0.0

        ai = np.zeros([Ni]).astype(np.float32)
        bi = np.zeros([Ni]).astype(np.float32)

        for ix in range(0, Ni):
            if (ix < WCPML):
                Fi = ((WCPML - 1 - ix) * di)
                d_i = d0 * velMax * (Fi / Li) * (Fi / Li)
                Alphai = np.pi * frequency * ((Li - Fi) / Li)
                bi[ix] = np.exp(-(d_i + Alphai) * dt)
                ai[ix] = (d_i / (d_i + Alphai)) * (bi[ix] - 1.0)

            if (ix >= WCPML and ix < (Ni - WCPML)):
                Fi = 0.0
                bi[ix] = 0.0
                ai[ix] = 0.0

            if (ix >= (Ni - WCPML) and ix < Ni):
                Fi = ((ix - (Ni - WCPML)) * di)
                d_i = d0 * velMax * (Fi / Li) * (Fi / Li)
                Alphai = np.pi * frequency * (Li - Fi) / Li

                bi[ix] = np.exp(-(d_i + Alphai) * dt)
                ai[ix] = (d_i / (d_i + Alphai)) * (bi[ix] - 1.0)

        return ai, bi

class StressTensor():
    ''' Mxx, Myy, Mzz, Mxy, Mxz, Myz '''

    def __init__(self, Mxx, Myy, Mzz, Mxy, Mxz, Myz):
        ''' Mxx, Myy, Mzz, Mxy, Mxz, Myz '''
        self.xx = Mxx
        self.yy = Myy
        self.zz = Mzz
        self.xy = Mxy
        self.xz = Mxz
        self.yz = Myz

    def __str__(self):
        return f"Mxx: {self.xx:.3f} Myy: {self.yy:.3f} Mzz: {self.zz:.3f} " \
               f"Mxy: {self.xy:.3f} Mxz: {self.xz:.3f} Myz: {self.yz:.3f}"


class Gradients():
    def __init__(self, mode, ny, nz, nx):
        self.Mxx = 0.0
        self.Myy = 0.0
        self.Mzz = 0.0
        self.Mxy = 0.0
        self.Mxz = 0.0
        self.Myz = 0.0
        
        if mode == 1:
            self.Lambda = np.zeros((ny, nz, nx)).astype(np.float32)
            self.Mu = np.zeros((ny, nz, nx)).astype(np.float32)
            self.Rho = np.zeros((ny, nz, nx)).astype(np.float32)

            self.d_Lambda = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
            self.d_Mu = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
            self.d_Rho = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))


    def setZero(self, mode, ny, nz, nx):
        self.Mxx = 0.0
        self.Myy = 0.0
        self.Mzz = 0.0
        self.Mxy = 0.0
        self.Mxz = 0.0
        self.Myz = 0.0
        
        if mode == 1:
            self.Lambda = np.zeros((ny, nz, nx)).astype(np.float32)
            self.Mu = np.zeros((ny, nz, nx)).astype(np.float32)
            self.Rho = np.zeros((ny, nz, nx)).astype(np.float32)

            self.d_Lambda = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
            self.d_Mu = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))
            self.d_Rho = gpuarray.to_gpu(np.zeros((ny, nz, nx)).astype(np.float32))

    def __str__(self):
        return f"gk_Mxx: {self.Mxx:.2f} gk_Myy: {self.Myy:.2f} gk_Mzz: {self.Mzz:.2f}" \
               f"gk_Mxy: {self.Mxy:.2f} gk_Mxz: {self.Mxz:.2f} gk_Myz: {self.Myz:.2f}"


class Elastic3D():
    def __init__(self, d_mk:Model, geometria:Geometry, ondicula:Wavelet, pml: CPML, mode=0):
        ''' mode:
        0: no guarda campos
        1: guarda campos Vx, Vy, Vz
        '''
        self.mk = Model(d_mk.nz, d_mk.ny, d_mk.nx, d_mk.dz, d_mk.dy, d_mk.dx)
        self.mk.Mu = d_mk.Mu.copy()
        self.mk.Lambda = d_mk.Lambda.copy()
        self.mk.Rho = d_mk.Rho.copy()
        self.mk.Vp = d_mk.Vp.copy()
        self.mk.Vs = d_mk.Vs.copy()
        
        self.nx = d_mk.nx
        self.ny = d_mk.ny
        self.nz = d_mk.nz
        self.dy = d_mk.dy
        self.dx = d_mk.dx
        self.dz = d_mk.dz
        self.geometria = geometria
        self.ondicula = ondicula
        self.cpml = pml
        self.nt = ondicula.nSamples
        self.forwFields = Fields(d_mk.ny, d_mk.nz, d_mk.nx, self.nt)
        self.backFields = Fields(d_mk.ny, d_mk.nz, d_mk.nx, self.nt)
        self.gk = Gradients(mode, self.ny, self.nz, self.nx)
        self.alpha = Alpha()
        self.costFunction = []
        self.mode = mode
        self.lenPML = pml.lenPML

    def setMode(self, mode):
        self.mode = mode


    def setIdxEarthquake(self, SrxIdx):
        self.SrxIdx = SrxIdx
        self.LenStations = self.geometria.sismos[SrcIdx].getLenStations()
        self.gather =  Gather(self.nt, self.geometria.sismos[SrcIdx].getLenStations())
        self.obsGather = Gather(self.nt, self.geometria.sismos[SrcIdx].getLenStations())
        self.modGather = Gather(self.nt, self.geometria.sismos[SrcIdx].getLenStations())
        self.resGather = Gather(self.nt, self.geometria.sismos[SrcIdx].getLenStations())
        self.gather.setZero()
        self.obsGather.setZero()
        self.modGather.setZero()
        self.resGather.setZero()
        
        self.rec_pos_x, self.rec_pos_y, self.rec_pos_z = self.geometria.getStationsEarthquake(SrxIdx)

        self.d_rec_pos_x = gpuarray.to_gpu(self.rec_pos_x).astype(np.int32)
        self.d_rec_pos_y = gpuarray.to_gpu(self.rec_pos_y).astype(np.int32)
        self.d_rec_pos_z = gpuarray.to_gpu(self.rec_pos_z).astype(np.int32)

        self.gg = gpuarray.to_gpu(np.zeros((6, 1)).astype(np.float32))

        self.Sx = int(self.geometria.sismos[SrxIdx].x)
        self.Sy = int(self.geometria.sismos[SrxIdx].y)
        self.Sz = int(self.geometria.sismos[SrxIdx].z)

    def compute_Txx_Tyy_Tzz_Txy_Txz_Tyz(self, d_mk: Model, d_field: Fields, flagMode = 2):
        ''' flagMode ->  1: from nt-1 to 0;  2: from 0 to nt-1'''
        # Esfuerzos Tauxx, Tauyy, Tauzz , Tauxy,  Tauxz, Tauyz
        kernel_Tau(d_field.Tauxx, d_field.Tauyy, d_field.Tauzz,
                   d_field.Tauxy, d_field.Tauxz, d_field.Tauyz,
                   d_field.Vx, d_field.Vy, d_field.Vz,
                   d_field.psidVx_dx, d_field.psidVy_dy, d_field.psidVz_dz,
                   d_field.psidVx_dz, d_field.psidVx_dy, d_field.psidVy_dx,
                   d_field.psidVy_dz, d_field.psidVz_dx, d_field.psidVz_dy,
                   self.cpml.d_ax, self.cpml.d_bx, self.cpml.d_ay, self.cpml.d_by, self.cpml.d_az, self.cpml.d_bz,
                   d_mk.d_Vp, d_mk.d_Vs, d_mk.d_Rho, d_mk.d_Lambda, d_mk.d_Mu,
                   np.float32(d_mk.dx), np.float32(d_mk.dy), np.float32(d_mk.dz), np.float32( (-1)**(flagMode)*self.ondicula.dt),
                   np.int32(d_mk.nx), np.int32(d_mk.ny), np.int32(d_mk.nz),
                   grid=(int(nx / threadBlockX) + 1, int(ny / threadBlockY) + 1, int(nz / threadBlockZ) + 1),
                   block=(threadBlockX, threadBlockY, threadBlockZ))

    def compute_Vx_Vy_Vz(self, d_mk: Model, d_field: Fields, flagMode = 2):
        # Campos Vx,  Vy, y Vz
        kernel_Vx_Vy_Vz(d_field.Vx, d_field.Vy, d_field.Vz,
                        d_field.Tauxx, d_field.Tauyy, d_field.Tauzz,
                        d_field.Tauxy, d_field.Tauxz, d_field.Tauyz,
                        d_field.psidTauxx_dx, d_field.psidTauzz_dz, d_field.psidTauyy_dy,
                        d_field.psidTauxy_dy, d_field.psidTauxy_dx, d_field.psidTauxz_dz,
                        d_field.psidTauxz_dx, d_field.psidTauyz_dz, d_field.psidTauyz_dy,
                        self.cpml.d_ax, self.cpml.d_bx, self.cpml.d_ay, self.cpml.d_by, self.cpml.d_az, self.cpml.d_bz,
                        d_mk.d_Rho,
                        np.float32(d_mk.dx), np.float32(d_mk.dy), np.float32(d_mk.dz), np.float32( (-1)**(flagMode)*self.ondicula.dt),
                        np.int32(d_mk.nx), np.int32(d_mk.ny), np.int32(d_mk.nz),
                        grid=(int(d_mk.nx / threadBlockX) + 1, int(d_mk.ny / threadBlockY) + 1,
                              int(d_mk.nz / threadBlockZ) + 1),
                        block=(threadBlockX, threadBlockY, threadBlockZ))
   
                                  
    def getBorder_Vx_Vy_Vz(self, d_field: Fields, it, WCPML):
        # get Vx
        
#        d_field.h_fVx[self.lenPML:self.ny-self.lenPML, :self.nz-self.lenPML, self.lenPML:self.nx-self.lenPML,it] = d_field.Vx.get()[self.lenPML:self.ny-self.lenPML, :self.nz-self.lenPML, self.lenPML:self.nx-self.lenPML]
#        d_field.h_fVy[self.lenPML:self.ny-self.lenPML, :self.nz-self.lenPML, self.lenPML:self.nx-self.lenPML,it] = d_field.Vy.get()[self.lenPML:self.ny-self.lenPML, :self.nz-self.lenPML, self.lenPML:self.nx-self.lenPML]
#        d_field.h_fVz[self.lenPML:self.ny-self.lenPML, :self.nz-self.lenPML, self.lenPML:self.nx-self.lenPML,it] = d_field.Vz.get()[self.lenPML:self.ny-self.lenPML, :self.nz-self.lenPML, self.lenPML:self.nx-self.lenPML]
        

        kernel_get_border(d_field.Vx, 
                          d_field.VxBorders.front, d_field.VxBorders.back, 
                          d_field.VxBorders.left, d_field.VxBorders.right,
                          d_field.VxBorders.top, d_field.VxBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))

        
        # get Vy
        kernel_get_border(d_field.Vy, 
                          d_field.VyBorders.front, d_field.VyBorders.back, 
                          d_field.VyBorders.left, d_field.VyBorders.right,
                          d_field.VyBorders.top, d_field.VyBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))               
         # get Vz
        kernel_get_border(d_field.Vz, 
                          d_field.VzBorders.front, d_field.VzBorders.back, 
                          d_field.VzBorders.left, d_field.VzBorders.right,
                          d_field.VzBorders.top, d_field.VzBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
 
    def getBorder_Tau(self, d_field: Fields, it, WCPML):
        # get Tauxx
        kernel_get_border(d_field.Tauxx, 
                          d_field.TauxxBorders.front, d_field.TauxxBorders.back, 
                          d_field.TauxxBorders.left, d_field.TauxxBorders.right,
                          d_field.TauxxBorders.top, d_field.TauxxBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
        
        kernel_get_border(d_field.Tauyy, 
                          d_field.TauyyBorders.front, d_field.TauyyBorders.back, 
                          d_field.TauyyBorders.left, d_field.TauyyBorders.right,
                          d_field.TauyyBorders.top, d_field.TauyyBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
    
        kernel_get_border(d_field.Tauzz, 
                          d_field.TauzzBorders.front, d_field.TauzzBorders.back, 
                          d_field.TauzzBorders.left, d_field.TauzzBorders.right,
                          d_field.TauzzBorders.top, d_field.TauzzBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
                          
        kernel_get_border(d_field.Tauxy, 
                          d_field.TauxyBorders.front, d_field.TauxyBorders.back, 
                          d_field.TauxyBorders.left, d_field.TauxyBorders.right,
                          d_field.TauxyBorders.top, d_field.TauxyBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
                          
                          
        kernel_get_border(d_field.Tauxz, 
                          d_field.TauxzBorders.front, d_field.TauxzBorders.back, 
                          d_field.TauxzBorders.left, d_field.TauxzBorders.right,
                          d_field.TauxzBorders.top, d_field.TauxzBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
                          
                          
        kernel_get_border(d_field.Tauyz, 
                          d_field.TauyzBorders.front, d_field.TauyzBorders.back, 
                          d_field.TauyzBorders.left, d_field.TauyzBorders.right,
                          d_field.TauyzBorders.top, d_field.TauyzBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
                          
    def setBorder_Tau(self, d_field: Fields, it, WCPML):
        # set Tauxx
        kernel_set_border(d_field.Tauxx, 
                          d_field.TauxxBorders.front, d_field.TauxxBorders.back, 
                          d_field.TauxxBorders.left, d_field.TauxxBorders.right,
                          d_field.TauxxBorders.top, d_field.TauxxBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
        
        kernel_set_border(d_field.Tauyy, 
                          d_field.TauyyBorders.front, d_field.TauyyBorders.back, 
                          d_field.TauyyBorders.left, d_field.TauyyBorders.right,
                          d_field.TauyyBorders.top, d_field.TauyyBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
    
        kernel_set_border(d_field.Tauzz, 
                          d_field.TauzzBorders.front, d_field.TauzzBorders.back, 
                          d_field.TauzzBorders.left, d_field.TauzzBorders.right,
                          d_field.TauzzBorders.top, d_field.TauzzBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
                          
        kernel_set_border(d_field.Tauxy, 
                          d_field.TauxyBorders.front, d_field.TauxyBorders.back, 
                          d_field.TauxyBorders.left, d_field.TauxyBorders.right,
                          d_field.TauxyBorders.top, d_field.TauxyBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
                          
                          
        kernel_set_border(d_field.Tauxz, 
                          d_field.TauxzBorders.front, d_field.TauxzBorders.back, 
                          d_field.TauxzBorders.left, d_field.TauxzBorders.right,
                          d_field.TauxzBorders.top, d_field.TauxzBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
                          
                          
        kernel_set_border(d_field.Tauyz, 
                          d_field.TauyzBorders.front, d_field.TauyzBorders.back, 
                          d_field.TauyzBorders.left, d_field.TauyzBorders.right,
                          d_field.TauyzBorders.top, d_field.TauyzBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
                          
    def setBorder_Vx_Vy_Vz(self, d_field: Fields, it, WCPML):
        
        

        kernel_set_border(d_field.Vx, 
                          d_field.VxBorders.front, d_field.VxBorders.back, 
                          d_field.VxBorders.left, d_field.VxBorders.right,
                          d_field.VxBorders.top, d_field.VxBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))

        
        # get Vy
        kernel_set_border(d_field.Vy, 
                          d_field.VyBorders.front, d_field.VyBorders.back, 
                          d_field.VyBorders.left, d_field.VyBorders.right,
                          d_field.VyBorders.top, d_field.VyBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))               
         # get Vz
        kernel_set_border(d_field.Vz, 
                          d_field.VzBorders.front, d_field.VzBorders.back, 
                          d_field.VzBorders.left, d_field.VzBorders.right,
                          d_field.VzBorders.top, d_field.VzBorders.bottom,
                          np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), np.int32(it),  np.int32(WCPML),
                          grid=(int(self.nx / threadBlockX) + 1, 
                                int(self.ny / threadBlockY) + 1,
                                int(self.nz / threadBlockZ) + 1),
                          block=(threadBlockX, threadBlockY, threadBlockZ))
                          
    def addSource(self, it, flagMode=2):
        self.forwFields.Tauxx[self.Sy, self.Sz, self.Sx] += (-1)**(flagMode)*self.M.xx * self.ondicula.waveform[it]
        self.forwFields.Tauyy[self.Sy, self.Sz, self.Sx] += (-1)**(flagMode)*self.M.yy * self.ondicula.waveform[it]
        self.forwFields.Tauzz[self.Sy, self.Sz, self.Sx] += (-1)**(flagMode)*self.M.zz * self.ondicula.waveform[it]
        self.forwFields.Tauxy[self.Sy, self.Sz, self.Sx] += (-1)**(flagMode)*self.M.xy * self.ondicula.waveform[it]
        self.forwFields.Tauxz[self.Sy, self.Sz, self.Sx] += (-1)**(flagMode)*self.M.xz * self.ondicula.waveform[it]
        self.forwFields.Tauyz[self.Sy, self.Sz, self.Sx] += (-1)**(flagMode)*self.M.yz * self.ondicula.waveform[it]



    def getStation(self, it):

        kernel_get_Station(self.forwFields.Vx, self.gather.d_Vx,
                           self.forwFields.Vy, self.gather.d_Vy,
                           self.forwFields.Vz, self.gather.d_Vz,
                           self.d_rec_pos_x, self.d_rec_pos_y,  self.d_rec_pos_z,
                           np.int32(self.LenStations), np.int32(self.ondicula.nSamples), np.int32(it),
                           np.int32(self.mk.nx), np.int32(self.mk.nz),
                           grid=(int(self.LenStations / threadBlockX) + 1, 1, 1),
                           block=(threadBlockX, 1, 1))


    def saveForwardFieldsPng(self, it):
        if it % 25 == 0:
            if self.cont == 0:
               self.Vclip = [np.amax(self.forwFields.h_Vx), np.amin(self.forwFields.h_Vx)]

            self.cont += 1

            plt.figure(figsize=(12, 12))
            plt.subplot(2, 2, 1)
            plt.imshow(self.forwFields.h_Vx[self.Sy, :, :], vmax=1e4 * self.Vclip[0], vmin=1e4 * self.Vclip[1])
            plt.colorbar()
            plt.ylabel('Z (Km)', size=18)
            plt.xlabel('X (Km)', size=18)
            plt.title(f'Corte Y ite:{it}', size=18)

            plt.subplot(2, 2, 3)
            plt.imshow(self.forwFields.h_Vx[:, self.Sz, :], vmax=1e4 * self.Vclip[0], vmin=1e4 * self.Vclip[1])
            plt.colorbar()
            plt.ylabel('Y (Km)', size=18)
            plt.xlabel('X (Km)', size=18)
            plt.title(f'Corte Z ite:{it}', size=18)

            plt.subplot(2, 2, 2)
            plt.imshow(self.forwFields.h_Vx[:, :, self.Sx], vmax=1e4 * self.Vclip[0], vmin=1e4 * self.Vclip[1])
            plt.colorbar()
            plt.ylabel('Y (Km)', size=18)
            plt.xlabel('Z (Km)', size=18)
            plt.title(f'Corte X ite:{it}', size=18)
            plt.savefig('Imagenes/Vx_' + str(self.cont) + '.png')
            plt.close()

    def savebackfieldsPng(self, it):
        if it % 25 == 0:
            if self.cont == 0:
               self.Vclip = [np.amax(self.backFields.h_Vx), np.amin(self.backFields.h_Vx)]
               print(self.Vclip[0], self.Vclip[1])
            self.cont += 1

            plt.figure(figsize=(12, 12))
            plt.subplot(2, 2, 1)
            plt.imshow(self.backFields.h_Vx[self.Sy, :, :],  vmax=1e-14, vmin=-1e-14 )
            plt.colorbar()
            plt.ylabel('Z (Km)', size=18)
            plt.xlabel('X (Km)', size=18)
            plt.title(f'Corte Y ite:{it}', size=18)

            plt.subplot(2, 2, 3)
            plt.imshow(self.backFields.h_Vx[:, self.Sz, :], vmax=1e-14, vmin=-1e-14 )
            plt.colorbar()
            plt.ylabel('Y (Km)', size=18)
            plt.xlabel('X (Km)', size=18)
            plt.title(f'Corte Z ite:{it}', size=18)

            plt.subplot(2, 2, 2)
            plt.imshow(self.backFields.h_Vx[:, :, self.Sx], vmax=1e-14, vmin=-1e-14 )
            plt.colorbar()
            plt.ylabel('Y (Km)', size=18)
            plt.xlabel('Z (Km)', size=18)
            plt.title(f'Corte X ite:{it}', size=18)
            plt.savefig('Imagenes/lambdaVx_' + str(self.cont) + '.png')
            plt.close()

    def forwardFields(self, strname='mod', saveFieldFlag = False):
        ''' setZero forward fields '''
        self.forwFields.setZero()

        ''' setZero gather '''
        self.gather.setZero()

        if strname == 'mod' and self.mode == 1:
            self.forwFields.hostMemAllocFields(self.ny, self.nz, self.nx, self.nt)

        self.cont = 0
        for it in (range(self.nt)):
            # 1. Esfuerzos Tauxx, Tauyy, Tauzz , Tauxy,  Tauxz, Tauyz
            self.compute_Txx_Tyy_Tzz_Txy_Txz_Tyz(self.mk, self.forwFields)
            # 2. Agregar esfuerzos Tauxx, Tauyy, Tauzz , Tauxy,  Tauxz, Tauyz
            self.addSource(it)
            # 3. Campos Vx,  Vy, y Vz
            self.compute_Vx_Vy_Vz(self.mk, self.forwFields)
            # 4. Obtener gathers Vx,  Vy, y Vz
            self.getStation(it)
            # 5. Guardar campos Vx,  Vy, y Vz
            if saveFieldFlag:
                # 5a. 
                self.saveForwardFieldsPng(it)
            if strname == 'mod' and self.mode == 1:
                # 5b. get borders Vx, Vy, Vz
                self.getBorder_Vx_Vy_Vz(self.forwFields, it, self.lenPML)
#                self.getBorder_Tau(self.forwFields, it, self.lenPML)

            
            
        if strname == 'mod':
            self.modGather.Vx, self.modGather.Vy, self.modGather.Vz = self.getGathers()

        if strname == 'obs':
            self.obsGather.Vx, self.obsGather.Vy, self.obsGather.Vz = self.getGathers()


    def addAdjointSource(self, it):

        kernel_add_adjoint_source(self.backFields.Vx, self.resGather.d_Vx,
                                  self.backFields.Vy, self.resGather.d_Vy,
                                  self.backFields.Vz, self.resGather.d_Vz,
                                  self.d_rec_pos_x, self.d_rec_pos_y, self.d_rec_pos_z,
                                  np.int32(self.LenStations), np.int32(self.ondicula.nSamples), np.int32(it),
                                  np.int32(self.mk.nx), np.int32(self.mk.nz),
                                  grid=(int(self.LenStations / threadBlockX) + 1, 1, 1),
                                  block=(threadBlockX, 1, 1))

    def computeGradients_Rho(self, it):

        kernel_Gradients_Rho(self.backFields.Vx, self.backFields.Vy, self.backFields.Vz,
                             self.forwFields.Vx, self.forwFields.Vy, self.forwFields.Vz,
                             self.forwFields.Vx_present, self.forwFields.Vy_present, self.forwFields.Vz_present,
                             self.mk.d_Vp, self.mk.d_Vs, self.mk.d_Rho,
                             self.gk.d_Rho,
                             np.int32(self.cpml.lenPML), np.int32(self.ondicula.nSamples), np.int32(it), np.float32(self.ondicula.dt),
                             np.int32(self.nx),  np.int32(self.ny),  np.int32(self.nz),
                             grid=(int(self.nx / threadBlockX) + 1, int(self.ny / threadBlockY) + 1, int(self.nz / threadBlockZ) + 1),
                             block=(threadBlockX, threadBlockY, threadBlockZ))
                             
        self.gk.Rho += self.gk.d_Rho.get()

    def computeGradients_Lambda(self, it):

        kernel_Gradients_Lambda(self.backFields.Tauxx, self.backFields.Tauyy, self.backFields.Tauzz,
                             self.forwFields.Vx_present, self.forwFields.Vy_present, self.forwFields.Vz_present,
                             self.mk.d_Vp, self.mk.d_Vs, self.mk.d_Rho, self.mk.d_Lambda, self.mk.d_Mu,
                             self.gk.d_Lambda, np.float32(self.mk.dx), np.float32(self.mk.dy), np.float32(self.mk.dz),
                             np.int32(self.cpml.lenPML), np.int32(self.ondicula.nSamples), np.int32(it), np.float32(self.ondicula.dt),
                             np.int32(self.mk.nx), np.int32(self.mk.ny),  np.int32(self.mk.nz),
                             grid=(int(self.nx / threadBlockX) + 1, int(self.ny / threadBlockY) + 1, int(self.nz / threadBlockZ) + 1),
                             block=(threadBlockX, threadBlockY, threadBlockZ))

        self.gk.Lambda += self.gk.d_Lambda.get()

    def computeGradients_Mu(self, it):

        kernel_Gradients_Mu(self.backFields.Tauxx, self.backFields.Tauyy, self.backFields.Tauzz,
                            self.backFields.Tauxy, self.backFields.Tauxz, self.backFields.Tauyz,
                            self.forwFields.Vx_present, self.forwFields.Vy_present, self.forwFields.Vz_present,
                            self.mk.d_Vp, self.mk.d_Vs, self.mk.d_Rho, self.mk.d_Lambda, self.mk.d_Mu,
                            self.gk.d_Mu, np.float32(self.mk.dx), np.float32(self.mk.dy), np.float32(self.mk.dz),
                            np.int32(self.cpml.lenPML), np.int32(self.ondicula.nSamples), np.int32(it), np.float32(self.ondicula.dt),
                            np.int32(self.mk.nx), np.int32(self.mk.ny), np.int32(self.mk.nz),
                            grid=(int(self.nx / threadBlockX) + 1, int(self.ny / threadBlockY) + 1, int(self.nz / threadBlockZ) + 1),
                             block=(threadBlockX, threadBlockY, threadBlockZ))

        self.gk.Mu += self.gk.d_Mu.get()

    def computeGradients_Mij(self, it):

        kernel_Gradients_Mij(self.backFields.Tauxx, self.backFields.Tauyy, self.backFields.Tauzz,
                             self.backFields.Tauxy, self.backFields.Tauxz, self.backFields.Tauyz,
                             self.ondicula.d_waveform,
                             self.mk.d_Vp, self.mk.d_Vs, self.mk.d_Rho, self.mk.d_Lambda, self.mk.d_Mu,
                             self.gg,
                             np.int32(self.Sx), np.int32(self.Sy), np.int32(self.Sz),
                             np.int32(self.ondicula.nSamples), np.int32(it),
                             np.int32(self.mk.nx), np.int32(self.mk.nz),
                             grid=(1, 1, 1),
                             block=(threadBlockX, 1, 1))

        self.gk.Mxx += float(self.gg.get()[0])
        self.gk.Myy += float(self.gg.get()[1])
        self.gk.Mzz += float(self.gg.get()[2])
        self.gk.Mxy += float(self.gg.get()[3])
        self.gk.Mxz += float(self.gg.get()[4])
        self.gk.Myz += float(self.gg.get()[5])
    
    @staticmethod
    def cudaMemCopy_DTD(self, d_out, d_in, ny, nz, nx):
         kernel_DTD_copy(d_out, d_in,
                        np.int32(nx), np.int32(ny), np.int32(nz),
                        grid=(int(nx / threadBlockX) + 1, int(ny / threadBlockY) + 1, int(nz / threadBlockZ) + 1),
                        block=(threadBlockX, threadBlockY, threadBlockZ))
    
    def backwardFields_FowardFieldsRemake(self, saveFieldFlag = False):
        ''' setZero backward fields '''
        self.backFields.setZero()

        ''' setZero gradients '''
        self.gk.setZero(self.mode, self.ny, self.nz, self.nx)

        
               
                
        for it in (range(0, self.ondicula.nSamples-1)):

            
            ''' Backpropagation '''
            # 1. Esfuerzos Tauxx, Tauyy, Tauzz , Tauxy,  Tauxz, Tauyz
            self.compute_Txx_Tyy_Tzz_Txy_Txz_Tyz(self.mk, self.backFields)
            # 2. Campos Vx,  Vy, y Vz
            self.compute_Vx_Vy_Vz(self.mk, self.backFields)
            # 3. Add adjoint source Vx,  Vy, y Vz
            self.addAdjointSource(it)


            ''' Remake forward fields''' 
            
            Elastic3D.cudaMemCopy_DTD(self, self.forwFields.Vx_present, self.forwFields.Vx, self.ny, self.nz, self.nx)
            Elastic3D.cudaMemCopy_DTD(self, self.forwFields.Vy_present, self.forwFields.Vy, self.ny, self.nz, self.nx)
            Elastic3D.cudaMemCopy_DTD(self, self.forwFields.Vz_present, self.forwFields.Vz, self.ny, self.nz, self.nx)
             
            # 1. Campos Vx,  Vy, y Vz en tiempo reverso
            self.compute_Vx_Vy_Vz(self.mk, self.forwFields,  flagMode = 1)
            # 2. Establecer bordes Vx,  Vy, y Vz
            self.setBorder_Vx_Vy_Vz(self.forwFields, self.nt-it-2, self.lenPML)
            # 3. Establecer cero CPML Vx,  Vy, Vz, Tau
            self.forwFields.setZeroCPMLzone(self.ny, self.nz, self.nx, self.lenPML)
            
            # 4. Esfuerzos Tauxx, Tauyy, Tauzz , Tauxy,  Tauxz, Tauyz
            self.compute_Txx_Tyy_Tzz_Txy_Txz_Tyz(self.mk, self.forwFields, flagMode = 1 )
            # 5. Agregar esfuerzos Tauxx, Tauyy, Tauzz , Tauxy,  Tauxz, Tauyz
            self.addSource(self.nt-it-1, flagMode=1)
            

#            if rank ==0 and it%50==0: 
                
                
#                Fields.graficar(self, self.forwFields.Vx_present, 'Vx', self.Sx, self.Sy, self.Sz)
#                Fields.graficar(self, self.backFields.Vx, 'back Vx', self.Sx, self.Sy, self.Sz)
#                Fields.graficar(self, self.backFields.Vy, 'back Vy', self.Sx, self.Sy, self.Sz)
#                Fields.graficar(self, self.backFields.Vz, 'back Vz', self.Sx, self.Sy, self.Sz)
#                
#                Fields.graficar(self, self.forwFields.Vx, 'Vx', self.Sx, self.Sy, self.Sz)
#                Fields.graficar(self, self.forwFields.Vy, 'Vy', self.Sx, self.Sy, self.Sz)
#                Fields.graficar(self, self.forwFields.Vz, 'Vz', self.Sx, self.Sy, self.Sz)
                 
#                plt.show(block=False)
#                plt.pause(2.0)
#                plt.close()
#                 Vclip = [np.amax(self.forwFields.VxBorders.right.get()), np.amin(self.forwFields.VxBorders.right.get())]
  
#                f = plt.figure(figsize=(20, 12))
#                plt.subplot(3, 3, 1)
#                plt.imshow(self.forwFields.Vx.get()[:, :, self.Sx], aspect="auto", vmax=1e-2 * Vclip[0], vmin=1e-2 * Vclip[1])
#                plt.title(f'it:{it} Corte x - Vx')
#                plt.ylabel('Y (Km)', size=18)
#                plt.xlabel('Z (Km)', size=18)
#                plt.colorbar()
#                plt.subplot(3, 3, 2)
#                plt.imshow(self.forwFields.Vx.get()[:, self.Sz, :], aspect="auto", vmax=1e-2 * Vclip[0], vmin=1e-2 * Vclip[1])
#                plt.title(f'it:{it} Corte z - Vx')
#                plt.ylabel('Y (Km)', size=18)
#                plt.xlabel('X (Km)', size=18)
#                plt.colorbar()
#                plt.subplot(3, 3, 3)
#                plt.imshow(self.forwFields.Vx.get()[self.Sy, :, :], aspect="auto", vmax=1e-2 * Vclip[0], vmin=1e-2 * Vclip[1])
#                plt.title(f'it:{it} Corte y - Vx')
#                plt.ylabel('Z (Km)', size=18)
#                plt.xlabel('X (Km)', size=18)
#                plt.colorbar()
#                plt.subplot(3, 3, 4)
#                plt.imshow(self.forwFields.h_fVx[:,:, self.Sx, self.nt-it-2], aspect="auto", vmax=1e-2 * Vclip[0], vmin=1e-2 * Vclip[1])
#                plt.ylabel('Y (Km)', size=18)
#                plt.xlabel('Z (Km)', size=18)
#                plt.colorbar()
#                plt.subplot(3, 3, 5)
#                plt.imshow(self.forwFields.h_fVx[:, self.Sz, :, self.nt-it-2], aspect="auto", vmax=1e-2 * Vclip[0], vmin=1e-2 * Vclip[1])
#                plt.ylabel('Y (Km)', size=18)
#                plt.xlabel('X (Km)', size=18)
#                plt.colorbar()
#                plt.subplot(3, 3, 6)
#                plt.imshow(self.forwFields.h_fVx[self.Sy, :, :, self.nt-it-2], aspect="auto", vmax=1e-2 * Vclip[0], vmin=1e-2 * Vclip[1])
#                plt.ylabel('Z (Km)', size=18)
#                plt.xlabel('X (Km)', size=18)
#                plt.colorbar()
#                plt.subplot(3, 3, 7)
#                plt.imshow(self.forwFields.Vx.get()[:, :, self.Sx]-self.forwFields.h_fVx[:,:, self.Sx, self.nt-it-2], aspect="auto")
#                plt.ylabel('Y (Km)', size=18)
#                plt.xlabel('Z (Km)', size=18)
#                plt.colorbar()
#                plt.subplot(3, 3, 8)
#                plt.imshow(self.forwFields.Vx.get()[:, self.Sz, :]-self.forwFields.h_fVx[:, self.Sz, :, self.nt-it-2], aspect="auto")
#                plt.ylabel('Y (Km)', size=18)
#                plt.xlabel('X (Km)', size=18)
#                plt.colorbar()
#                plt.subplot(3, 3, 9)
#                plt.imshow(self.forwFields.Vx.get()[self.Sy, :, :]-self.forwFields.h_fVx[self.Sy, :, :, self.nt-it-2], aspect="auto")
#                plt.ylabel('Z (Km)', size=18)
#                plt.xlabel('X (Km)', size=18)
#                plt.colorbar()
#                if it == 250 or it == 500:
#                    f.savefig(f"pdf/Actual/Vx_comparison_it_{it}.pdf", bbox_inches='tight')
   
#                plt.show(block=False)
#                plt.pause(2.0)
#                plt.close()
                
            # 4. Compute Gradients
            if self.mode == 0:
                self.computeGradients_Mij(it)
            
            elif self.mode == 1 and it < self.nt-1:
                self.computeGradients_Rho(it)
                self.computeGradients_Lambda(it)
                self.computeGradients_Mu(it)
                
            # 5. Guardar campos adjuntos Vx,  Vy, y Vz
            if saveFieldFlag:
                self.savebackfieldsPng(it)
                
    def backwardFields(self, saveFieldFlag = False):
        ''' setZero forward fields '''
        self.backFields.setZero()

        ''' setZero gradients '''
        self.gk.setZero(self.mode, self.ny, self.nz, self.nx)

        self.cont = 0
        for it in (range(self.ondicula.nSamples)):
            # 1. Esfuerzos Tauxx, Tauyy, Tauzz , Tauxy,  Tauxz, Tauyz
            self.compute_Txx_Tyy_Tzz_Txy_Txz_Tyz(self.mk, self.backFields)
            # 2. Campos Vx,  Vy, y Vz
            self.compute_Vx_Vy_Vz(self.mk, self.backFields)
            # 3. Add adjoint source Vx,  Vy, y Vz
            self.addAdjointSource(it)
            # 4. Compute Gradients
            if self.mode == 0:
                self.computeGradients_Mij(it)
            
            elif self.mode == 1 and it < self.nt-1:
                self.computeGradients_Rho(it)
                self.computeGradients_Lambda(it)
                self.computeGradients_Mu(it)
                
            # 5. Guardar campos adjuntos Vx,  Vy, y Vz
            if saveFieldFlag:
                self.savebackfieldsPng(it)

    def computeAlpha(self):

        self.alpha.Mxx = 0.0 if self.gk.Mxx == 0 else 5e-2 / np.sqrt(np.sum(self.gk.Mxx**2))
        self.alpha.Myy = 0.0 if self.gk.Myy == 0 else 5e-2 / np.sqrt(np.sum(self.gk.Myy**2))
        self.alpha.Mzz = 0.0 if self.gk.Mzz == 0 else 5e-2 / np.sqrt(np.sum(self.gk.Mzz**2))
        self.alpha.Mxy = 0.0 if self.gk.Mxy == 0 else 5e-2 / np.sqrt(np.sum(self.gk.Mxy**2))
        self.alpha.Mxz = 0.0 if self.gk.Mxz == 0 else 5e-2 / np.sqrt(np.sum(self.gk.Mxz**2))
        self.alpha.Myz = 0.0 if self.gk.Myz == 0 else 5e-2 / np.sqrt(np.sum(self.gk.Myz**2))

#        print(self.alpha)

    def updateStressTensor(self):
        self.M.xx += float(self.alpha.Mxx * self.gk.Mxx)
        self.M.yy += float(self.alpha.Myy * self.gk.Myy)
        self.M.zz += float(self.alpha.Mzz * self.gk.Mzz)
        self.M.xy += float(0.5 * self.alpha.Mxy * self.gk.Mxy)
        self.M.xz += float(0.5 * self.alpha.Mxz * self.gk.Mxz)
        self.M.yz += float(0.5 * self.alpha.Myz * self.gk.Myz)

        # print(self.gk)
#        print(f"gkMxx:{self.gk.Mxx} gkMyy:{self.gk.Myy} gkMzz:{self.gk.Mzz} gkMxz:{self.gk.Mxz} gkMxy:{self.gk.Mxy} gkMyz:{self.gk.Myz}")
#        print(f"Mxx:{self.M.xx:.3f} Myy:{self.M.yy:.3f} Mzz:{self.M.zz:.3f} Mxz:{self.M.xz:.3f} Mxy:{self.M.xy:.3f} Myz:{self.M.yz:.3f}")


   
    def setStressTensor(self, M):
        self.M = copy.copy(M)
#        print(self.M)

    def setModel(self, model):
        self.mk.Mu = model.Mu.copy()
        self.mk.Lambda = model.Lambda.copy()
        self.mk.Rho = model.Rho.copy()
        self.mk.Vp = model.Vp.copy()
        self.mk.Vs = model.Vs.copy()
        self.mk.hostToDevice()
        
    
    def getGathers(self):
        return self.gather.d_Vx.get().copy(), self.gather.d_Vy.get().copy(), self.gather.d_Vz.get().copy()

    def getModel(self):
        return self.mk

    def computeResidual(self):
        self.resGather.Vx = self.modGather.Vx - self.obsGather.Vx
        self.resGather.Vy = self.modGather.Vy - self.obsGather.Vy
        self.resGather.Vz = self.modGather.Vz - self.obsGather.Vz

        ''' H2D  transfer'''
        self.resGather.hostToDevice()
        
    def computeCostFunction(self, iter):
        self.totalCostFunction = 0.5 * (np.sum((self.resGather.Vx) ** 2)
                                + np.sum((self.resGather.Vy) ** 2)
                                + np.sum((self.resGather.Vz) ** 2) )


        self.costFunction.append(float(self.totalCostFunction))


    def computeResidualCrossCorrelation(self):

        for i in range(self.LenStations):
            l2_dobsVx = np.sqrt(np.sum(self.obsGather.Vx[:, i] ** 2))
            l2_dobsVy = np.sqrt(np.sum(self.obsGather.Vy[:, i] ** 2))
            l2_dobsVz = np.sqrt(np.sum(self.obsGather.Vz[:, i] ** 2))
            l2_dmodVx = np.sqrt(np.sum(self.modGather.Vx[:, i] ** 2))
            l2_dmodVy = np.sqrt(np.sum(self.modGather.Vy[:, i] ** 2))
            l2_dmodVz = np.sqrt(np.sum(self.modGather.Vz[:, i] ** 2))
            dmod_dobs_Vx = np.sum(self.modGather.Vx[:, i] * self.obsGather.Vx[:, i])
            dmod_dobs_Vy = np.sum(self.modGather.Vy[:, i] * self.obsGather.Vy[:, i])
            dmod_dobs_Vz = np.sum(self.modGather.Vz[:, i] * self.obsGather.Vz[:, i])

            if l2_dobsVx == 0: l2_dobsVx = 1.0
            if l2_dobsVy == 0: l2_dobsVy = 1.0
            if l2_dobsVz == 0: l2_dobsVz = 1.0
            if l2_dmodVx == 0: l2_dmodVx = 1.0
            if l2_dmodVy == 0: l2_dmodVy = 1.0
            if l2_dmodVz == 0: l2_dmodVz = 1.0
            if dmod_dobs_Vx == 0: dmod_dobs_Vx = 1.0
            if dmod_dobs_Vy == 0: dmod_dobs_Vy = 1.0
            if dmod_dobs_Vz == 0: dmod_dobs_Vz = 1.0

            # self.resGather.Vx[:, i] = np.round(dmod_dobs_Vx / (l2_dmodVx * l2_dobsVx), decimals=8) * (self.modGather.Vx[:,i] - self.obsGather.Vx[:,i])
            # self.resGather.Vy[:, i] = np.round(dmod_dobs_Vy / (l2_dmodVy * l2_dobsVy), decimals=8) * (self.modGather.Vy[:,i] - self.obsGather.Vy[:,i])
            # self.resGather.Vz[:, i] = np.round(dmod_dobs_Vz / (l2_dmodVz * l2_dobsVz), decimals=8) * (self.modGather.Vz[:,i] - self.obsGather.Vz[:,i])

            # self.resGather.Vx[:, i] = (self.modGather.Vx[:, i] - self.obsGather.Vx[:, i])
            # self.resGather.Vy[:, i] = (self.modGather.Vy[:, i] - self.obsGather.Vy[:, i])
            # self.resGather.Vz[:, i] = (self.modGather.Vz[:, i] - self.obsGather.Vz[:, i])

            # self.resGather.Vx[:, i] = 2*(self.modGather.Vx[:, i] - self.obsGather.Vx[:, i])/dmod_dobs_Vx - np.sum((self.modGather.Vx[:, i] - self.obsGather.Vx[:, i])**2)*  self.obsGather.Vx[:, i]/(dmod_dobs_Vx**2)
            # self.resGather.Vy[:, i] = 2*(self.modGather.Vy[:, i] - self.obsGather.Vy[:, i])/dmod_dobs_Vy - np.sum((self.modGather.Vy[:, i] - self.obsGather.Vy[:, i])**2)*  self.obsGather.Vy[:, i]/(dmod_dobs_Vy**2)
            # self.resGather.Vz[:, i] = 2*(self.modGather.Vz[:, i] - self.obsGather.Vz[:, i])/dmod_dobs_Vz - np.sum((self.modGather.Vz[:, i] - self.obsGather.Vz[:, i])**2)*  self.obsGather.Vz[:, i]/(dmod_dobs_Vz**2)

            self.resGather.Vx[:, i] = (self.modGather.Vx[:, i]) * (dmod_dobs_Vx / (l2_dobsVx * (l2_dmodVx ** 3)) - (self.obsGather.Vx[:, i]) / (l2_dmodVx * l2_dobsVx))
            self.resGather.Vy[:, i] = (self.modGather.Vy[:, i]) * (dmod_dobs_Vy / (l2_dobsVy * (l2_dmodVy ** 3)) - (self.obsGather.Vy[:, i]) / (l2_dmodVy * l2_dobsVy))
            self.resGather.Vz[:, i] = (self.modGather.Vz[:, i]) * (dmod_dobs_Vz / (l2_dobsVz * (l2_dmodVz ** 3)) - (self.obsGather.Vz[:, i]) / (l2_dmodVz * l2_dobsVz))
    def computeCostFunctionCrossCorrelation(self, iter):

        l2_dobsVx = np.sqrt( np.sum(self.obsGather.Vx[:] * self.obsGather.Vx[:]) )
        l2_dobsVy = np.sqrt( np.sum(self.obsGather.Vy[:] * self.obsGather.Vy[:]) )
        l2_dobsVz = np.sqrt( np.sum(self.obsGather.Vz[:] * self.obsGather.Vz[:]) )
        l2_dmodVx = np.sqrt( np.sum(self.modGather.Vx[:] * self.modGather.Vx[:]) )
        l2_dmodVy = np.sqrt( np.sum(self.modGather.Vy[:] * self.modGather.Vy[:]) )
        l2_dmodVz = np.sqrt( np.sum(self.modGather.Vz[:] * self.modGather.Vz[:]) )
        dmod_dobs_Vx = np.sum( self.modGather.Vx[:] * self.obsGather.Vx[:])
        dmod_dobs_Vy = np.sum( self.modGather.Vy[:] * self.obsGather.Vy[:])
        dmod_dobs_Vz = np.sum( self.modGather.Vz[:] * self.obsGather.Vz[:])

        if l2_dobsVx==0: l2_dobsVx=1.0
        if l2_dobsVy==0: l2_dobsVy=1.0
        if l2_dobsVz==0: l2_dobsVz=1.0
        if l2_dmodVx==0: l2_dmodVx=1.0
        if l2_dmodVy==0: l2_dmodVy=1.0
        if l2_dmodVz==0: l2_dmodVz=1.0

        costVx = 1.0 - (dmod_dobs_Vx) / (l2_dmodVx * l2_dobsVx)
        costVy = 1.0 - (dmod_dobs_Vy) / (l2_dmodVy * l2_dobsVy)
        costVz = 1.0 - (dmod_dobs_Vz) / (l2_dmodVz * l2_dobsVz)

        self.totalCostFunction = (costVx + costVy + costVz)

        print(f'\ncostFunction[{iter}]:{self.totalCostFunction} ')
        self.costFunction.append(float(self.totalCostFunction))


    def graficarObs_Mod_Res_Gathers(self):
        self.obsGather.graficar('ObsGather')
        self.modGather.graficar('modGather')
        self.resGather.graficar('Residual')

    def fwi_Mij(self, nIte, flagSave = False, tolFWI = 1e-5):

        Mmodxx = [float(self.M.xx)]
        Mmodyy = [float(self.M.yy)]
        Mmodzz = [float(self.M.zz)]
        Mmodxy = [float(self.M.xy)]
        Mmodxz = [float(self.M.xz)]
        Mmodyz = [float(self.M.yz)]
        modData = Gather(self.ondicula.nSamples, self.geometria.sismos[self.SrxIdx].getLenStations())
        modData.setZero()
        
        for iter in range(nIte):
            if rank==0: print('\n|' + '-' * 10 + 'Ite ' + str(iter) + '-' * 10 + '|')
            comm.Barrier()            

            ''' Datos modelados '''
#            if rank ==0: print(f"\nDatos Modelados")
            self.setStressTensor(self.M)
#            inicio = time.time()
            self.forwardFields('mod')
#            fin = time.time()
#            print(f"Time propagation: {fin - inicio}")

            modData.Vx, modData.Vy, modData.Vz = self.getGathers()

            ''' Gather ini '''
            
            if iter == 0:
                save(f'Results/Actual/Gathers/modDataIni{self.SrxIdx}.Vx.npy', modData.Vx)
                save(f'Results/Actual/Gathers/modDataIni{self.SrxIdx}.Vy.npy', modData.Vy)
                save(f'Results/Actual/Gathers/modDataIni{self.SrxIdx}.Vz.npy', modData.Vz)
            ''' residual '''
            self.computeResidual()
            # self.computeResidualCrossCorrelation()

            '''6) Cost Function'''
            self.computeCostFunction(iter)
            # self.computeCostFunctionCrossCorrelation(iter)
            comm.Barrier()            
            print(f"{self.SrxIdx}-> {self.M}")
#            if rank ==0: print()
            print(f'{self.SrxIdx}-> costFunction[{iter}]:{self.totalCostFunction} ')
#            if rank ==0: print()
            ''' Graficar Gathers '''
            # self.graficarObs_Mod_Res_Gathers()

#            print(f"\nBackPropagation")
#            inicio = time.time()
            self.backwardFields()
            
#            fin = time.time()
#            print(f"Time nBackPropagation: {fin - inicio}")

            cM = copy.copy(self.M)
            cMtemp = copy.copy(self.M)
            
            
            if iter == 0:
                ''' Compute alpha '''
                self.computeAlpha()
                '''Actualizar Tensor de momentos'''
                self.updateStressTensor()
            else:
                cMtemp.xx = self.M.xx + self.alpha.Mxx * self.gk.Mxx
                cMtemp.yy = self.M.yy + self.alpha.Myy * self.gk.Myy
                cMtemp.zz = self.M.zz + self.alpha.Mzz * self.gk.Mzz
                cMtemp.xy = self.M.xy + 0.5 * self.alpha.Mxy * self.gk.Mxy
                cMtemp.xz = self.M.xz + 0.5 * self.alpha.Mxz * self.gk.Mxz
                cMtemp.yz = self.M.yz + 0.5 * self.alpha.Myz * self.gk.Myz

                self.setStressTensor(cMtemp)
                self.forwardFields('mod')
                self.computeResidual()
                
                
                totalCostFunction = 0.5 * (np.sum((self.resGather.Vx) ** 2) + np.sum((self.resGather.Vy) ** 2) + np.sum((self.resGather.Vz) ** 2) )


                if not(np.isnan(self.costFunction[iter])) and (totalCostFunction-self.costFunction[iter])< 0:
                   self.M.xx = cMtemp.xx 
                   self.M.yy = cMtemp.yy 
                   self.M.zz = cMtemp.zz 
                   self.M.xy = cMtemp.xy 
                   self.M.xz = cMtemp.xz 
                   self.M.yz = cMtemp.yz 
                   
                   
                else:
                   self.alpha.Mxx /= 2
                   self.alpha.Myy /= 2
                   self.alpha.Mzz /= 2
                   self.alpha.Mxy /= 2
                   self.alpha.Mxz /= 2
                   self.alpha.Myz /= 2
                        
                   self.setStressTensor(cM)

                        
            Mmodxx.append(self.M.xx)
            Mmodyy.append(self.M.yy)
            Mmodzz.append(self.M.zz)
            Mmodxy.append(self.M.xy)
            Mmodxz.append(self.M.xz)
            Mmodyz.append(self.M.yz)

            
            save(f'Results/Actual/CostFunction/costFunction{self.SrxIdx}.npy', self.costFunction)
            save(f'Results/Actual/TM/Mmodxx{self.SrxIdx}.npy', Mmodxx)
            save(f'Results/Actual/TM/Mmodyy{self.SrxIdx}.npy', Mmodyy)
            save(f'Results/Actual/TM/Mmodzz{self.SrxIdx}.npy', Mmodzz)
            save(f'Results/Actual/TM/Mmodxy{self.SrxIdx}.npy', Mmodxy)
            save(f'Results/Actual/TM/Mmodxz{self.SrxIdx}.npy', Mmodxz)
            save(f'Results/Actual/TM/Mmodyz{self.SrxIdx}.npy', Mmodyz)
            save(f'Results/Actual/Gathers/modData{self.SrxIdx}.Vx.npy', modData.Vx)
            save(f'Results/Actual/Gathers/modData{self.SrxIdx}.Vy.npy', modData.Vy)
            save(f'Results/Actual/Gathers/modData{self.SrxIdx}.Vz.npy', modData.Vz)
            save(f'Results/Actual/Gathers/resData{self.SrxIdx}.Vx.npy', self.resGather.Vx)
            save(f'Results/Actual/Gathers/resData{self.SrxIdx}.Vy.npy', self.resGather.Vy)
            save(f'Results/Actual/Gathers/resData{self.SrxIdx}.Vz.npy', self.resGather.Vz)
            
            # if iter > 0 and fabs(self.costFunction[iter]-self.costFunction[iter-1]) < tolFWI:
            #     break

#        plt.figure()
#        plt.plot(np.array(self.costFunction))
#        plt.xlabel('Iteraciones', size=18)
#        plt.ylabel('Funcion de Costo', size=18)
#        plt.show(block=False)
#        plt.pause(0.5)
#        plt.close()
        
        
    def fwi_Vp_Vs_Rho(self, data_obs, listaTM, nIter,  flagSave=False, tolFWI = 1e-5, lbfgsLayers=10, ite_cte=2):

        self.costFunction = []

        self.mk.Lambda = self.mk.d_Lambda.get().copy()
        self.mk.Mu = self.mk.d_Mu.get().copy()
        self.mk.Rho = self.mk.d_Rho.get().copy()

        # -------------------------------------------------------------------------------------------------------------#
        ''' Parametros LBFGS'''
        ykLambda = np.zeros((lbfgsLayers, self.ny,  self.nz, self.nx)).astype(np.float32)
        skLambda = np.zeros((lbfgsLayers, self.ny,  self.nz,self. nx)).astype(np.float32)
        ykMu = np.zeros((lbfgsLayers, self.ny, self.nz, self.nx)).astype(np.float32)
        skMu = np.zeros((lbfgsLayers, self.ny, self.nz, self.nx)).astype(np.float32)
        ykRho = np.zeros((lbfgsLayers, self.ny, self.nz, self.nx)).astype(np.float32)
        skRho = np.zeros((lbfgsLayers, self.ny, self.nz, self.nx)).astype(np.float32)
        # -------------------------------------------------------------------------------------------------------------#

        for iter in range(nIter):

            if rank == 0: print('\n|' + '-' * 10 + 'Ite ' + str(iter) + '-' * 10 + '|')
            tempRho = np.zeros((self.ny, self.nz, self.nx)).astype(np.float32)
            tempMu = np.zeros((self.ny, self.nz, self.nx)).astype(np.float32)
            tempLambda = np.zeros((self.ny, self.nz, self.nx)).astype(np.float32)
            self.totalCostFunction = 0
            modData = [None for x in range(nSismos)]

            t_Rho = np.zeros((sizeRank, self.ny, self.nz, self.nx))
            t_Mu = np.zeros((sizeRank, self.ny, self.nz, self.nx))
            t_Lambda = np.zeros((sizeRank, self.ny, self.nz, self.nx))

            inicio = time.time()


            for SrcIdx in (range(rank, nSismos, sizeRank)): # Loop over earthquakes

#                comm.Barrier()
                self.setStressTensor(listaTM[SrcIdx])
                ''' Set src index '''
                comm.Barrier()
                self.setIdxEarthquake(SrcIdx)
                ''' Datos observados '''
                self.obsGather.Vx = data_obs[SrcIdx].Vx.copy()
                self.obsGather.Vy = data_obs[SrcIdx].Vy.copy()
                self.obsGather.Vz = data_obs[SrcIdx].Vz.copy()

                ''' Datos modelados '''
                Data = Gather(self.ondicula.nSamples, self.geometria.sismos[SrcIdx].getLenStations())
                # print(f"\nPropagation")


                self.forwardFields('mod', saveFieldFlag=False)
                Data.Vx, Data.Vy, Data.Vz = self.getGathers()
#                modData.append(Data)
                modData[SrcIdx] = Data
                
                       
                '''------- Acumular obsData MPI ---------'''
                for irank in range(1, sizeRank):
                    if rank == irank:
                        comm.send(modData[SrcIdx], dest=0, tag=11)
                    elif rank == 0:
                        modData[SrcIdx+irank] = comm.recv(source=irank, tag=11)

                ''' Gather ini '''
                if iter == 0:
                    save('Results/Actual/Gathers/modData0.Vx.npy', Data.Vx)
                    save('Results/Actual/Gathers/modData0.Vy.npy', Data.Vy)
                    save('Results/Actual/Gathers/modData0.Vz.npy', Data.Vz)

                ''' residual '''
                self.computeResidual()

                '''6) Cost Function'''
                CostFunction = 0.5 * (np.sum(self.resGather.Vx ** 2)+ np.sum(self.resGather.Vy ** 2) + np.sum(self.resGather.Vz ** 2))
                
                tempCostFunction = comm.reduce( CostFunction, op=MPI.SUM, root=0)
                tempCostFunction= comm.bcast(tempCostFunction, root=0)
                self.totalCostFunction += tempCostFunction

                ''' Graficar Gathers '''
                # self.graficarObs_Mod_Res_Gathers()

                # print(f"\nBackPropagation")
#                self.backwardFields(saveFieldFlag=False)
                self.backwardFields_FowardFieldsRemake(saveFieldFlag=False)
                
#                ctx.pop() #deactivate again
#                ctx.detach() #delete it
#                exit()
                
                t_Rho[rank, :, :, :] = self.gk.Rho.copy()
                t_Lambda[rank, :, :, :] = self.gk.Lambda.copy()
                t_Mu[rank, :, :, :] = self.gk.Mu.copy()
                
#                tempRho += self.gk.Rho
#                tempMu += self.gk.Mu
#                tempLambda += self.gk.Lambda

#                comm.Barrier()


                '''7) Acumular Gradientes '''
                ''' Temp Rho gradient '''
                for irank in range(1, sizeRank):
                    if rank == irank:
                        comm.Send([t_Rho[irank, :, :, :], MPI.FLOAT], dest=0, tag=77)
                    elif rank == 0:
                        comm.Recv([t_Rho[irank, :, :, :], MPI.FLOAT], source=irank, tag=77)
                ''' Temp Lambda gradient '''
                for irank in range(1, sizeRank):
                    if rank == irank:
                        comm.Send([t_Lambda[irank, :, :, :], MPI.FLOAT], dest=0, tag=77)
                    elif rank == 0:
                        comm.Recv([t_Lambda[irank, :, :, :], MPI.FLOAT], source=irank, tag=77)
                 
                ''' Temp Mu gradient '''
                for irank in range(1, sizeRank):
                    if rank == irank:
                        comm.Send([t_Mu[irank, :, :, :], MPI.FLOAT], dest=0, tag=77)
                    elif rank == 0:
                        comm.Recv([t_Mu[irank, :, :, :], MPI.FLOAT], source=irank, tag=77)
                        
    
                tempRho += t_Rho.sum(axis=0) # revisar
                tempMu += t_Mu.sum(axis=0)
                tempLambda += t_Lambda.sum(axis=0)



            ''' Gaussian Filter Gradient '''
            tempRho = gaussian_filter(tempRho, sigma=1)
            tempMu = gaussian_filter(tempMu, sigma=1)
            tempLambda = gaussian_filter(tempLambda, sigma=1)
            
            tempRho[:, :10,:] = 0
            tempMu[:, :10, :] = 0
            tempLambda[:, :10, :] = 0
            
            tempRho[:self.lenPML+2, :,:] = 0
            tempRho[self.ny-self.lenPML-2:, :,:] = 0
            tempRho[:, :, :self.lenPML+2] = 0
            tempRho[:, :, self.nx-self.lenPML-2:] = 0
            tempRho[:, self.nz-self.lenPML-2:, :] = 0
                        
            tempMu[:self.lenPML+2, :,:] = 0
            tempMu[self.ny-self.lenPML-2:, :,:] = 0
            tempMu[:, :, :self.lenPML+2] = 0
            tempMu[:, :, self.nx-self.lenPML-2:] = 0
            tempMu[:, self.nz-self.lenPML-2:, :] = 0

            tempLambda[:self.lenPML+2, :,:] = 0
            tempLambda[self.ny-self.lenPML-2:, :,:] = 0
            tempLambda[:, :, :self.lenPML+2] = 0
            tempLambda[:, :, self.nx-self.lenPML-2:] = 0
            tempLambda[:, self.nz-self.lenPML-2:, :] = 0

                
            tempRho = comm.bcast(tempRho, root=0)
            tempMu = comm.bcast(tempMu, root=0)
            tempLambda = comm.bcast(tempLambda, root=0)
            
            if rank == 0:
                Model.graficar(tempLambda, 'tempLambda',  corte, corte, corte)
                Model.graficar(tempMu, 'tempMu',  corte, corte, corte)
                Model.graficar(tempRho, 'tempRho',  corte, corte, corte)
                
                plt.show()
      
#            ctx.pop() #deactivate again
#            ctx.detach() #delete it
#            exit()
            
            ''' save modData'''
            if rank == 0 : save(f'Results/Actual/Gathers/modData_{fq}Hz.npy', modData)

            ''' Compute alpha '''
            if rank == 0: print(f'\ncostFunction[{iter}]:{self.totalCostFunction} ')
            self.costFunction.append(float(self.totalCostFunction))
            # ---------------------------------------------------------------------------------------------------------#
            ''' Compute sk, yk'''
            compute_sk_yk(skLambda, ykLambda, self.mk.d_Lambda.get().copy(), tempLambda, lbfgsLayers, iter)
            compute_sk_yk(skMu, ykMu, self.mk.d_Mu.get().copy(), tempMu, lbfgsLayers, iter)
            compute_sk_yk(skRho, ykRho, self.mk.d_Rho.get().copy(), tempRho, lbfgsLayers, iter)
            # ---------------------------------------------------------------------------------------------------------#
            cLambda = self.mk.d_Lambda.get().copy()
            cMu = self.mk.d_Mu.get().copy()
            cRho = self.mk.d_Rho.get().copy()
                
            cLambda= comm.bcast(cLambda, root=0)
            cMu= comm.bcast(cMu, root=0)
            cRho= comm.bcast(cRho, root=0)
            # ---------------------------------------------------------------------------------------------------------#
            ''' L-BFGS'''
            if iter >= ite_cte and iter > 0:
                ''' Starting L-BFGS '''
                if rank == 0: print(f'Starting L-BFGS')

                optimalCost = 1e20
                nTry = 10
                rLambda = L_BFGS(skLambda, ykLambda, tempLambda, lbfgsLayers, iter)
                rMu = L_BFGS(skMu, ykMu, tempMu, lbfgsLayers, iter)
                rRho = L_BFGS(skRho, ykRho, tempRho, lbfgsLayers, iter)

                rLambda= comm.bcast(rLambda, root=0)
                rMu= comm.bcast(rMu, root=0)
                rRho= comm.bcast(rRho, root=0)
                
         

                alpha = 1
                intento = 0
                cLambda = self.mk.d_Lambda.get().copy()
                cMu = self.mk.d_Mu.get().copy()
                cRho = self.mk.d_Rho.get().copy()
                
                cLambda= comm.bcast(cLambda, root=0)
                cMu= comm.bcast(cMu, root=0)
                cRho= comm.bcast(cRho, root=0)
                
                if rank == 0: print(f'alpha:{alpha}')
                if rank == 0: print(f'alpha*rLambda: {alpha*np.amax(rLambda):.2e} alpha*rMu: {alpha*np.amax(rMu):.2e}  alpha*rRho: {alpha*np.amax(rRho):.2e}')
                if rank == 0: print(f'Max Min self.mk.Mu: {np.amax(self.mk.Mu):.2e} {np.amin(self.mk.Mu):.2e}')
                if rank == 0: print(f'Max Min self.mk.Lambda: {np.amax(self.mk.Lambda):.2e} {np.amin(self.mk.Lambda):.2e}')
                if rank == 0: print(f'Max Min self.mk.rho: {np.amax(self.mk.Rho):.2e} {np.amin(self.mk.Rho):.2e}')

                while (optimalCost > self.costFunction[iter] or np.isnan(optimalCost)):
                    intento += 1
                    alpha /= 2
                    self.mk.Lambda = cLambda - alpha * rLambda
                    self.mk.Mu = cMu - alpha * rMu
#                    self.mk.Rho = cRho - alpha/5 * rRho
                    
                    self.mk.compute_Vp_Vs()
                    self.mk.compute_Rho()
                    self.mk.Rho[:, :10, :] = cRho[:, :10, :]

                    self.mk.Lambda = comm.bcast(self.mk.Lambda, root=0)
                    self.mk.Mu = comm.bcast(self.mk.Mu, root=0)
                    self.mk.Rho = comm.bcast(self.mk.Rho, root=0)
                  
                   
                    self.mk.hostToDevice()

                    totalCostFunction = 0.0

                    for SrcIdx in (range(rank, nSismos, sizeRank)): # Loop over earthquakes
                        

                        self.setStressTensor(listaTM[SrcIdx])
                        self.setIdxEarthquake(SrcIdx)
                        ''' Datos observados '''
                        self.obsGather.Vx = data_obs[SrcIdx].Vx.copy()
                        self.obsGather.Vy = data_obs[SrcIdx].Vy.copy()
                        self.obsGather.Vz = data_obs[SrcIdx].Vz.copy()

                        ''' Datos modelados '''
                        modData = Gather(self.ondicula.nSamples, self.geometria.sismos[SrcIdx].getLenStations())
                        self.forwardFields('mod', saveFieldFlag=False)
                        modData.Vx, modData.Vy, modData.Vz = self.getGathers()


                        ''' Residual '''
                        self.computeResidual()
                        
                        CostFunction = 0.5 * (np.sum(self.resGather.Vx ** 2)+ np.sum(self.resGather.Vy ** 2) + np.sum(self.resGather.Vz ** 2))

                        tempCostFunction = comm.reduce( CostFunction, op=MPI.SUM, root=0)
                        tempCostFunction= comm.bcast(tempCostFunction, root=0)
                        totalCostFunction += tempCostFunction
                        
                    optimalCost = totalCostFunction

                    if rank == 0: print(f'\ntry={intento} optimalCost[{iter}]:{optimalCost} ')
                    if np.isnan(optimalCost) :
                        self.mk.Lambda = cLambda
                        self.mk.Mu = cMu
                        self.mk.Rho = cRho
                        
                        
                        self.mk.Lambda = comm.bcast(self.mk.Lambda, root=0)
                        self.mk.Mu = comm.bcast(self.mk.Mu, root=0)
                        self.mk.Rho = comm.bcast(self.mk.Rho, root=0)
                        
                        self.mk.hostToDevice()
                        if intento >= nTry:
                            break
                        else:
                            continue
                    elif intento >= nTry:
                        self.mk.Lambda = cLambda
                        self.mk.Mu = cMu
                        self.mk.Rho = cRho
                        
                        self.mk.Lambda = comm.bcast(self.mk.Lambda, root=0)
                        self.mk.Mu = comm.bcast(self.mk.Mu, root=0)
                        self.mk.Rho = comm.bcast(self.mk.Rho, root=0)   

                        self.mk.hostToDevice()
                        break
                else:
                    self.mk.Lambda = cLambda - alpha * rLambda
                    self.mk.Mu = cMu - alpha * rMu
#                    self.mk.Rho = cRho - alpha/5 * rRho
                    self.mk.compute_Vp_Vs()
                    self.mk.compute_Rho()
                    self.mk.Rho[:, :10, :] = cRho[:, :10, :]

                    self.costFunction[iter] = optimalCost
                    
                    
                    self.mk.Lambda = comm.bcast(self.mk.Lambda, root=0)
                    self.mk.Mu = comm.bcast(self.mk.Mu, root=0)
                    self.mk.Rho = comm.bcast(self.mk.Rho, root=0)
                    
                    self.mk.hostToDevice()

            else:
                '''11) Update model with alpha constant'''

                self.alpha.Rho = 1e-6 * np.sqrt(np.sum(np.ravel(self.mk.Rho) ** 2)) / np.amax(tempRho)
                self.alpha.Mu = 1e-4 * np.sqrt(np.sum(np.ravel(self.mk.Mu) ** 2)) / np.amax(tempMu)
                self.alpha.Lambda = 1e-5 * np.sqrt(np.sum(np.ravel(self.mk.Lambda) ** 2)) / np.amax(tempLambda)
                
#                self.alpha.Rho = 1e-6 * np.sqrt(np.sum(np.ravel(self.mk.Rho) ** 2)) / np.amax(tempRho)
#                self.alpha.Mu = 1e-6 * np.sqrt(np.sum(np.ravel(self.mk.Mu) ** 2)) / np.amax(tempMu)
#                self.alpha.Lambda = 1e-7 * np.sqrt(np.sum(np.ravel(self.mk.Lambda) ** 2)) / np.amax(tempLambda)
#               
                if rank==0: print(f'\nalphaRho: {self.alpha.Rho:.2e} alphaLambda: {self.alpha.Lambda:.2e}  alphaMu: {self.alpha.Mu:.2e}')
                if rank==0: print(f'alphaRho*tempRho: {self.alpha.Rho*np.amax(tempRho):.2e} alphaLambda*tempLambda: {self.alpha.Lambda*np.amax(tempLambda):.2e}  alphaMu*tempMu: {self.alpha.Mu*np.amax(tempMu):.2e}')
                if rank==0: print(f'Max Min self.mk.Mu: {np.amax(self.mk.Mu):.2e} {np.amin(self.mk.Mu):.2e}')
                if rank==0: print(f'Max Min self.mk.Lambda: {np.amax(self.mk.Lambda):.2e} {np.amin(self.mk.Lambda):.2e}')
                if rank==0: print(f'Max Min self.mk.rho: {np.amax(self.mk.Rho):.2e} {np.amin(self.mk.Rho):.2e}')
#                    print()
#                    print(np.amax(tempLambda), np.amin(tempLambda))
#                    print(np.linalg.norm(np.ravel(tempLambda), ord=2), np.sqrt(np.sum(np.ravel(self.mk.Lambda) ** 2)))
#                    print(self.alpha.Lambda, self.alpha.Mu, self.alpha.Rho)


                '''Actualizar Mu, Lambda, Rho'''

                self.mk.Lambda += self.alpha.Lambda * tempLambda
                self.mk.Mu += self.alpha.Mu * tempMu
#                self.mk.Rho += self.alpha.Rho * tempRho
                 
                self.mk.compute_Vp_Vs()
                self.mk.compute_Rho()
                self.mk.Rho[:, :10, :] = cRho[:, :10, :]
    
                self.mk.Lambda = comm.bcast(self.mk.Lambda, root=0)
                self.mk.Mu = comm.bcast(self.mk.Mu, root=0)
                self.mk.Rho = comm.bcast(self.mk.Rho, root=0)
                
                self.mk.hostToDevice()

            # self.mk.compute_Vp_Vs()
            # print(f"prueba: {(self.costFunction[iter] - self.costFunction[iter - 1])}")
            # if iter > 0 and (self.costFunction[iter] - self.costFunction[iter - 1]) > 0:
            #     break

            fin = time.time()
            if rank == 0: print("Tiempo de iteracion:", fin - inicio)
            if rank == 0: save(f'Results/Actual/Models/Vp_{fq}Hz.npy', self.mk.Vp)
            if rank == 0: save(f'Results/Actual/Models/Vs_{fq}Hz.npy', self.mk.Vs)
            if rank == 0: save(f'Results/Actual/Models/Rho_{fq}Hz.npy', self.mk.Rho)
            if rank == 0: save(f'Results/Actual/Models/Lambda_{fq}Hz.npy', self.mk.Lambda)
            if rank == 0: save(f'Results/Actual/Models/Mu_{fq}Hz.npy', self.mk.Mu)
            if rank == 0: save(f'Results/Actual/CostFunction/costFunction_{fq}Hz.npy', self.costFunction)
            if rank == 0: save(f'Results/Actual/Gradients/gRho_{fq}Hz.npy', tempRho)
            if rank == 0: save(f'Results/Actual/Gradients/gMu_{fq}Hz.npy', tempMu)
            if rank == 0: save(f'Results/Actual/Gradients/gLambda_{fq}Hz.npy', tempLambda)



class Alpha():
    def __init__(self):
        self.Mxx = 0.0
        self.Myy = 0.0
        self.Mzz = 0.0
        self.Mxy = 0.0
        self.Mxz = 0.0
        self.Myz = 0.0

    def __str__(self):
        return f"alpha.Mxx: {self.Mxx:.4f} alpha.Myy: {self.Myy:.4f} alpha.Mzz: {self.Mzz:.4f} \nalpha.Mxy: {self.Mxy:.4f} alpha.Mxz: {self.Mxz:.4f} alpha.Myz: {self.Myz:.4f}"


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------------------------------------#
    N_gpu = drv.Device(0).count()
    if rank ==0: print(f"Numero de GPUS disponibles:{N_gpu}")
    # -----------------------------------------------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#
    ''' 1. Definiciones iniciales '''
    nx = 100                     # Dimension x
    ny = 100                     # Dimension y
    nz = 100                     # Dimension z
    dx = 1e3                     # Discretizacion en x (m)
    dy = dx                      # Discretizacion en y (m)
    dz = dx                      # Discretizacion en z (m)
    multFq = [0.1, 0.2, 0.3]     # Frecuencia central
    nSismos = 24                  # Numero de sismos
    lenPML = 20                  # Ancho de la CPML
    tprop = 120.0                # Tiempo de propagacion en segundos
    nIteTM = 2000                 # Numero de iteraciones de FWI
    nIteFWI = 30
    tolFWI = 1e-5                # Minima toleracia permitida en FWI
    nRxLine = 20                 # Numero de estaciones por linea
    nLinesRx = 10                # Numero de lineas
    nRec = nRxLine * nLinesRx    # Numero de estaciones
    corte = 50
    mode = 2
    H_STENCIL = 2
    # -----------------------------------------------------------------------------------------------------------------#
    ''' 2. Definiciones Threads/block GPU'''
    threadBlockX = 32          # Numero de threads en x por block
    threadBlockY = 2           # Numero de threads en y por block
    threadBlockZ = 4           # Numero de threads en z por block

    # -----------------------------------------------------------------------------------------------------------------#
    ''' 3. Definir tensor de esfuerzos dato observado '''
#    Mobs = StressTensor(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    Mobs = StressTensor(1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    save('Results/Actual/TM/targetTM.npy', np.array([Mobs.xx, Mobs.yy,  Mobs.zz, Mobs.xy, Mobs.xz, Mobs.yz]))
    # -----------------------------------------------------------------------------------------------------------------#
    ''' 4. Definir tensor de esfuerzos dato modelado '''
    # Mmod = StressTensor(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
#    Mmod = StressTensor(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    Mmod = StressTensor(0.5, 0.0, 0.5, 0.0, 0.5, 0.0)
    # Mmod = StressTensor(np.random.rand(), np.random.rand(),np.random.rand(),
    #                     np.random.rand(), np.random.rand(), np.random.rand())
    # -----------------------------------------------------------------------------------------------------------------#
    ''' 5. Crear los parametros mTarget (Vp, Vs, Rho)'''
    mTarget = Model(nz, ny, nx, dz, dy, dx)
    
    ''' Leer datos mTarget(Vp, Vs, Rho) '''
    mTarget.Rho = np.load('npyFiles/rhoTrue.npy')[:ny, :nz, :nx]*1e3
    mTarget.Vp = np.load('npyFiles/vpTrue.npy')[:ny, :nz, :nx]
    mTarget.Vs = np.load('npyFiles/vsTrue.npy')[:ny, :nz, :nx]
    
    mTarget.Vp[:, :10,:] = 2800.0
    mTarget.Vs[:, :10,:] = 1600.0
    mTarget.Rho[:, :10,:] = 2e3
#    
#    mTarget.Vp[:, :, :] = 4.5e3
#    mTarget.Vs[:, :, :] = 0.0
#    mTarget.Rho[:, :, :] = 2e3
     
    mTarget.compute_Lambda_Mu()
    mTarget.MaxMinVp()

    ''' 5.a Transferencia desde CPU a GPU '''
    mTarget.hostToDevice()

    ''' 5.b Graficar modelos: Vp, Vs, rho, mu, lambda'''
    if rank == 0: 
        Model.graficar(mTarget.Vp, "Vp", corte, corte, corte)
        Model.graficar(mTarget.Vs, "Vs", corte, corte, corte)
        Model.graficar(mTarget.Rho, "rho", corte, corte, corte)
        Model.graficar(mTarget.Lambda, "\lambda", corte, corte, corte)
        Model.graficar(mTarget.Mu, "\mu", corte, corte, corte)
        
        save('Results/Actual/Models/VpTarget.npy', mTarget.Vp)
        save('Results/Actual/Models/VsTarget.npy', mTarget.Vs)
        save('Results/Actual/Models/RhoTarget.npy', mTarget.Rho)
        save('Results/Actual/Models/LambdaTarget.npy', mTarget.Lambda)
        save('Results/Actual/Models/MuTarget.npy', mTarget.Mu)
    
#    ctx.pop() #deactivate again
#    ctx.detach() #delete it
#    exit()
    # -----------------------------------------------------------------------------------------------------------------#
    ''' 8. Geometria '''
    ''' 8.a Posiciones de Sismos '''
    src_pos_x = np.random.randint(lenPML, nx - lenPML, size=nSismos, dtype=int)
    src_pos_y = np.random.randint(lenPML, ny - lenPML, size=nSismos, dtype=int)
#    src_pos_z = np.random.randint(nz - 2*lenPML, nz - lenPML, size=nSismos, dtype=int)
    src_pos_z = np.ones(nSismos).astype(int) * (2)

    
#    path = 'FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite'
#    src_pos_x = np.load(f'Results/{path}/Geometry/sismos_pos_x.npy')
#    src_pos_y = np.load(f'Results/{path}/Geometry/sismos_pos_y.npy')
#    src_pos_z = np.load(f'Results/{path}/Geometry/sismos_pos_z.npy')
#    
    src_pos_y[nSismos-8:] = 52
    
#    src_pos_x = src_pos_x[-8:]
#    src_pos_y = src_pos_y[-8:]
#    src_pos_z = src_pos_z[-8:]
        
    src_pos_x = comm.bcast(src_pos_x , root=0)
    src_pos_y = comm.bcast(src_pos_y , root=0)
    src_pos_z = comm.bcast(src_pos_z , root=0)
    
    nSismos = len(src_pos_x)
    
#    print(src_pos_x)
#    print(src_pos_y)
#    print(src_pos_z)
    
    ''' 8.b Posiciones de Estaciones '''
    xx = np.linspace(lenPML, nx - lenPML-1, nRxLine).astype(int)
    yy = np.linspace(lenPML, ny - lenPML-1, nLinesRx).astype(int)

    rec_pos_x = np.array([]).astype(int)
    rec_pos_y = np.array([]).astype(int)
    rec_pos_z = np.ones(nRec).astype(int) * (2)

    for j in range(nLinesRx):
        for i in range(nRxLine):
            rec_pos_x = np.append(rec_pos_x, xx[i])
            rec_pos_y = np.append(rec_pos_y, yy[j])

#    rec_pos_x = rec_pos_x[:20]
#    rec_pos_y = [52 for x in range(len(rec_pos_x))]
#    rec_pos_z = [2 for x in range(len(rec_pos_x))]
    ''' Cargar las posiciones de las estaciones del proyecto '''

#    rec_pos_x = (load('LecturaEstacionesSismologicasProyecto/npyFiles/rec_pos_x_SGC_temporal_permanente.npy')/(2e3/dx)).astype(int)
#    rec_pos_y = (load('LecturaEstacionesSismologicasProyecto/npyFiles/rec_pos_y_SGC_temporal_permanente.npy')/(2e3/dy)).astype(int)

#    rec_pos_x = (load('LecturaEstacionesSismologicasProyecto/npyFiles/rec_pos_x_SGC.npy')/(2e3/dx)).astype(int)
#    rec_pos_y = (load('LecturaEstacionesSismologicasProyecto/npyFiles/rec_pos_y_SGC.npy')/(2e3/dy)).astype(int)
    
#    rec_pos_x = (load('LecturaEstacionesSismologicasProyecto/npyFiles/rec_pos_x_temporal.npy')/(2e3/dx)).astype(int)
#    rec_pos_y = (load('LecturaEstacionesSismologicasProyecto/npyFiles/rec_pos_y_temporal.npy')/(2e3/dy)).astype(int)
    
    
#    rec_pos_z = np.ones(len(rec_pos_x)).astype(int)*(2)
    
    ''' 8.c Crear objeto Geometry '''
    geometry = Geometry()

    for idx in range(nSismos):
        geometry.addSismo(Earthquake(src_pos_x[idx], src_pos_y[idx], src_pos_z[idx]))

    for idnSismos in range(nSismos):
        for idnStations in range(len(rec_pos_x)):
            geometry.sismos[idnSismos].addStation(Station(rec_pos_x[idnStations], rec_pos_y[idnStations], rec_pos_z[idnStations]))

    if rank==0:geometry.printSismos()
    if rank==0:geometry.printStations(0)
    if rank==0: geometry.graficar()
    if rank==0: geometry.save()
    
  
    if mode == 2:
        # -----------------------------------------------------------------------------------------------------------------#
        # -----------------------------------------------------------------------------------------------------------------#
        fq = multFq[0]
        ''' 6.Crear objeto Wavelet  '''
        wavelet = Wavelet(fq, tprop, mTarget)
        wavelet.ricker()
        # wavelet.gaussian()
        if rank==0: wavelet.graficar()
    
        # -----------------------------------------------------------------------------------------------------------------#
        ''' 7. Crear objeto CMPL '''
        cpml = CPML(lenPML, wavelet, mTarget)
        if rank==0: cpml.graficar()
        # -----------------------------------------------------------------------------------------------------------------#
        
        # -----------------------------------------------------------------------------------------------------------------#
        ''' 9. Crear datos observados '''
        obsData = [None for x in range(nSismos)]
    
    #    obsData = Gather(wavelet.nSamples, geometry.sismos[SrcIdx].getLenStations())
    
        ''' 9. Crear datos observados '''
        if rank== 0: print(f"\nDatos Observados")
    
        objElastic3D = Elastic3D(mTarget, geometry, wavelet, cpml, mode=mode)
        objElastic3D.setStressTensor(Mobs)
        objElastic3D.setModel(mTarget)
        
        
        for SrcIdx in tqdm(range(rank, nSismos, sizeRank)): # Loop over earthquakes
            comm.Barrier()
            Data = Gather(wavelet.nSamples, geometry.sismos[0].getLenStations())
            Data.setZero()
    #        inicio = time.time()
            objElastic3D.setIdxEarthquake(SrcIdx)
                    
#            print(f"\nSismo {SrcIdx}")
#            print(f"Sx:{objElastic3D.Sx} Sy:{objElastic3D.Sy} Sz:{objElastic3D.Sz}")
            objElastic3D.forwardFields('obs')
            Data.Vx, Data.Vy, Data.Vz = objElastic3D.getGathers()
            obsData[SrcIdx] = Data
            
    #        fin = time.time()
    #        print("Tiempo de propagación:", fin - inicio)
                    
            '''------- Acumular obsData MPI ---------'''
            for irank in range(1, sizeRank):
                if rank == irank:
                    comm.send(obsData[SrcIdx], dest=0, tag=11)
                elif rank == 0:
                    obsData[SrcIdx+irank] = comm.recv(source=irank, tag=11)
                    
            
        if rank == 0:        
            for SrcIdx in tqdm(range(0, nSismos)): # Loop over earthquakes
                obsData[SrcIdx].graficar()
        
        if rank == 0: save(f'Results/Actual/Gathers/obsData_{fq}Hz.npy', obsData)
        if rank == 0: save(f'Results/Actual/Gathers/obsData3D_{fq}Hz_Vpcte_Vscte_Rhocte.npy', obsData)
        
        
#        ctx.pop() #deactivate again
#        ctx.detach() #delete it
#        exit()
        # -----------------------------------------------------------------------------------------------------------------#
        ''' 10. Starting FWI '''
        cn = Model(nz, ny, nx, dz, dy, dx)
    
        cn.Vp = gaussian_filter(mTarget.Vp , sigma=10)
        cn.Vs = gaussian_filter(mTarget.Vs , sigma=10)
        cn.Rho = gaussian_filter(mTarget.Rho , sigma=10)
        cn.Vp[:, :10, :]= mTarget.Vp[:, :10, :]
        cn.Vs[:, :10, :] = mTarget.Vs[:, :10, :]
        cn.Rho[:, :10, :] = mTarget.Rho[:, :10, :]
        cn.compute_Lambda_Mu()
        cn.hostToDevice()
        
        '''============================================================ '''
        ''' Cargar Vp, Vs, Rho ''' 
#        path = "FWI_Lambda_Mu_RhoNafeDrake_Seismology_Nsrc25_TM_est_400ite"
#        cn.Mu = np.load(f'Results/{path}/Models/Mu_0.3Hz.npy')
#        cn.Lambda = np.load(f'Results/{path}/Models/Lambda_0.3Hz.npy')
#        cn.Rho = np.load(f'Results/{path}/Models/Rho_0.3Hz.npy')
#        
#        cn.Lambda = comm.bcast(cn.Lambda, root=0)
#        cn.Mu = comm.bcast(cn.Mu, root=0)
#        cn.Rho = comm.bcast(cn.Rho, root=0)
#        
#        cn.hostToDevice()
        '''============================================================ '''
        
        if rank == 0: 
            Model.graficar(cn.Vp, "Vp", corte, corte, corte)
            Model.graficar(cn.Vs, "Vs", corte, corte, corte)
            Model.graficar(cn.Rho, "rho", corte, corte, corte)
            Model.graficar(cn.Lambda, "\lambda", corte, corte, corte)
            Model.graficar(cn.Mu, "\mu", corte, corte, corte)
            plt.show()
    
        
        if rank == 0: save('Results/Actual/Models/VpInicial.npy', cn.Vp)
        if rank == 0: save('Results/Actual/Models/VsInicial.npy', cn.Vs)
        if rank == 0: save('Results/Actual/Models/RhoInicial.npy', cn.Rho)
        if rank == 0: save('Results/Actual/Models/LambdaInicial.npy', cn.Lambda)
        if rank == 0: save('Results/Actual/Models/MuInicial.npy', cn.Mu)
        
        ''' Crear lista de Tensores de momento '''
        listTM = []
        for iTensor in range(nSismos):
            listTM.append(Mmod)

        ''' Cargar Tensores '''
#        if rank == 0: listTM = load('Results/Inversion_TM_Target_Mxx_1.0_Myy_1.0_Mzz_1.0_Mxy_0.0_Mxz_0.0_Myz_0.0/TM/listTM_0.1Hz.npy', allow_pickle=True)
#        listTM = comm.bcast(listTM, root=0)

        if rank == 0:
            print('-'*10)
            print('\nTensores Iniciales:')
            for src in range(nSismos):
                print(f"Earthquake {src} -> {listTM[src]}")

        
        ''' 10. Starting Mij '''
   
        # -----------------------------------------------------------------------------------------------------------------#
    #    '''Fijando Rho, Lambda y Mu '''
    #    objElastic3D.mk.Lambda = mTarget.Lambda.copy()
    #    objElastic3D.mk.Mu = mTarget.Mu.copy()
    #    objElastic3D.mk.Rho = mTarget.Rho.copy()
    #    objElastic3D.mk.d_Lambda = gpuarray.to_gpu(mTarget.Lambda.copy())
    #    objElastic3D.mk.d_Mu = gpuarray.to_gpu(mTarget.Mu.copy())
    #    objElastic3D.mk.d_Rho = gpuarray.to_gpu(mTarget.Rho.copy())
        
        # -----------------------------------------------------------------------------------------------------------------#
        for SrcIdx in tqdm(range(rank, nSismos, sizeRank)): # Loop over earthquakes
            objElastic3D = Elastic3D(cn, geometry, wavelet, cpml, mode=0)
            objElastic3D.setModel(cn)
            objElastic3D.setMode(0)
            objElastic3D.setIdxEarthquake(SrcIdx)
            
            objElastic3D.setStressTensor(listTM[SrcIdx])
            objElastic3D.obsGather.Vx = obsData[SrcIdx].Vx.copy()
            objElastic3D.obsGather.Vy = obsData[SrcIdx].Vy.copy()
            objElastic3D.obsGather.Vz = obsData[SrcIdx].Vz.copy()
            save(f'Results/Actual/Gathers/obsData{SrcIdx}.Vx.npy', obsData[SrcIdx].Vx)
            save(f'Results/Actual/Gathers/obsData{SrcIdx}.Vy.npy', obsData[SrcIdx].Vy)
            save(f'Results/Actual/Gathers/obsData{SrcIdx}.Vz.npy', obsData[SrcIdx].Vz)
            objElastic3D.fwi_Mij(nIteTM)
    
            listTM[SrcIdx] = copy.copy(objElastic3D.M)
                    
            '''------- Acumular MPI ---------'''
            for irank in range(1, sizeRank):
                if rank == irank:
                    comm.send(listTM[SrcIdx], dest=0, tag=11)
                elif rank == 0:
                    listTM[SrcIdx+irank] = comm.recv(source=irank, tag=11)
            
        
        if rank == 0:
            print('-'*10)
            print('\nTensores:')
            for src in range(nSismos):
                print(f"Earthquake {src} -> {listTM[src]}")

                
        if rank == 0: save(f'Results/Actual/TM/listTM_{fq}Hz.npy', listTM)
                
        ctx.pop() #deactivate again
        ctx.detach() #delete it
                
        exit()
          
    # -----------------------------------------------------------------------------------------------------------------#
    elif mode == 1:
        ''' inversion Vp, Vs, Rho'''
        # -------------------------------------------------------------------------------------------------------------#
        cn = Model(nz, ny, nx, dz, dy, dx)
    
        cn.Vp = gaussian_filter(mTarget.Vp.copy() , sigma=10)
        cn.Vs = gaussian_filter(mTarget.Vs.copy() , sigma=10)
        cn.Rho = gaussian_filter(mTarget.Rho.copy() , sigma=10)
        cn.Vp[:, :10, :]= mTarget.Vp[:, :10, :]
        cn.Vs[:, :10, :] = mTarget.Vs[:, :10, :]
        cn.compute_Rho()
        cn.Rho[:, :10, :] = mTarget.Rho[:, :10, :]
        cn.compute_Lambda_Mu()
        cn.hostToDevice()
        
        if rank == 0: save('Results/Actual/Models/VpInicial.npy', cn.Vp)
        if rank == 0: save('Results/Actual/Models/VsInicial.npy', cn.Vs)
        if rank == 0: save('Results/Actual/Models/RhoInicial.npy', cn.Rho)
        if rank == 0: save('Results/Actual/Models/LambdaInicial.npy', cn.Lambda)
        if rank == 0: save('Results/Actual/Models/MuInicial.npy', cn.Mu)
        
        
        # -------------------------------------------------------------------------------------------------------------#
        '''---------------------------- '''
        ''' Cargar Vp, Vs, Rho ''' 
#        path = "Actual"
#        cn.Mu = np.load(f'Results/{path}/Models/Mu_0.1Hz.npy')
#        cn.Lambda = np.load(f'Results/{path}/Models/Lambda_0.1Hz.npy')
#        cn.Rho = np.load(f'Results/{path}/Models/Rho_0.1Hz.npy')
#        cn.hostToDevice()
        '''---------------------------- '''
                
        if rank == 0: 
            Model.graficar(cn.Vp, "Vp", corte, corte, corte)
            Model.graficar(cn.Vs, "Vs", corte, corte, corte)
            Model.graficar(cn.Rho, "rho", corte, corte, corte)
            Model.graficar(cn.Lambda, "\lambda", corte, corte, corte)
            Model.graficar(cn.Mu, "\mu", corte, corte, corte)
            plt.show()
      
        ''' Crear lista de tensores o cargarlos'''
        listTM = []
        for iTensor in range(nSismos):
            listTM.append(Mobs)
#            listTM.append(Mmod)
        
        ''' Cargar Tensores '''
        
#        if rank == 0: listTM = load('Results/Inversion_TM_Target_Mxx_1.0_Myy_1.0_Mzz_1.0_Mxy_0.0_Mxz_0.0_Myz_0.0_TM3D_2000ite_8Sismos/TM/listTM_0.1Hz.npy', allow_pickle=True)
#        if rank == 0: listTM = load('Results/Inversion_TM_Target_Mxx_1.0_Myy_0.0_Mzz_1.0_Mxy_0.0_Mxz_0.0_Myz_0.0_TM3D_2000ite_8Sismos/TM/listTM_0.1Hz.npy', allow_pickle=True)
#        if rank == 0: listTM = load('Results/Inversion_TM_Target_Mxx_1.0_Myy_0.0_Mzz_1.0_Mxy_0.0_Mxz_0.0_Myz_0.0_TM3D_2000ite_24Sismos/TM/listTM_0.1Hz.npy', allow_pickle=True)

        listTM = comm.bcast(listTM, root=0)
        
        
        if rank == 0:
            print('-'*10)
            print('\nTensores Iniciales:')
            for src in range(nSismos):
                print(f"Earthquake {src} -> {listTM[src]}")


        ''' Inicio FWI '''                
        for fq in multFq[0:]:
            if rank == 0: print(f"frequency: {fq} Hz")
            cn.MaxMinVp()

            # ---------------------------------------------------------------------------------------------------------#
            ''' 6.Crear objeto Wavelet  '''
            wavelet = Wavelet(fq, tprop, mTarget)
            wavelet.ricker()
            wavelet.graficar()
            # ---------------------------------------------------------------------------------------------------------#
            ''' 8. Crear objeto CMPL '''
            cpml = CPML(lenPML, wavelet, mTarget)
            cpml.graficar()
            # ---------------------------------------------------------------------------------------------------------#
        
            ''' 9. Crear datos observados '''
                
            objElastic3D = Elastic3D(mTarget, geometry, wavelet, cpml, mode=mode)
            objElastic3D.setStressTensor(Mobs)
            objElastic3D.setModel(mTarget)
            
           
            obsData = [None for x in range(nSismos)]
            

            # ---------------------------------------------------------------------------------------------------------#
            for SrcIdx in tqdm(range(rank, nSismos, sizeRank)): # Loop over earthquakes

                Data = Gather(wavelet.nSamples, geometry.sismos[0].getLenStations())
                Data.setZero()
                
                objElastic3D.setIdxEarthquake(SrcIdx)
                
                print(f"\nSismo {SrcIdx}")
                print(f"Sx:{objElastic3D.Sx} Sy:{objElastic3D.Sy} Sz:{objElastic3D.Sz}")
                objElastic3D.forwardFields('obs')
                Data.Vx, Data.Vy, Data.Vz = objElastic3D.getGathers()
                obsData[SrcIdx] = copy.copy(Data)
                

#                fin = time.time()
#                if rank == 0: print("Tiempo de propagación:", fin - inicio)
                '''------- Acumular MPI ---------'''
                for irank in range(1, sizeRank):
                    if rank == irank:
                        comm.send(obsData[SrcIdx], dest=0, tag=11)
                    elif rank == 0:
                        obsData[SrcIdx+irank] = comm.recv(source=irank, tag=11)
            
            if rank == 0:        
                for SrcIdx in tqdm(range(0, nSismos)): # Loop over earthquakes
                    obsData[SrcIdx].graficar()

            if rank ==0: save(f'Results/Actual/Gathers/obsData_{fq}Hz.npy', obsData)
            
   
            # -------------------------------------------------------------------------------------------------------- #
            objElastic3D = Elastic3D(cn, geometry, wavelet, cpml, mode=mode)
            objElastic3D.setModel(cn)
            # -------------------------------------------------------------------------------------------------------- #
            '''Fijando Rho, Lambda y Mu '''
#            objElastic3D.mk.Lambda = mTarget.Lambda.copy()
#            objElastic3D.mk.Mu = mTarget.Mu.copy()
#            objElastic3D.mk.Rho = mTarget.Rho.copy()
#            objElastic3D.mk.d_Lambda = gpuarray.to_gpu(mTarget.Lambda.copy())
#            objElastic3D.mk.d_Mu = gpuarray.to_gpu(mTarget.Mu.copy())
#            objElastic3D.mk.d_Rho = gpuarray.to_gpu(mTarget.Rho.copy())
            # -------------------------------------------------------------------------------------------------------- #
            ''' 11. Starting FWI Vp, Vs, Rho'''
            objElastic3D.fwi_Vp_Vs_Rho(obsData, listTM, nIteFWI)
            # -------------------------------------------------------------------------------------------------------- #
            cn.Rho = objElastic3D.mk.Rho.copy()
            cn.Lambda = objElastic3D.mk.Lambda.copy()
            cn.Mu = objElastic3D.mk.Mu.copy()
            
            
            cn.Lambda = comm.bcast(cn.Lambda, root=0)
            cn.Mu = comm.bcast(cn.Mu, root=0)
            cn.Rho = comm.bcast(cn.Rho, root=0)
            
            cn.hostToDevice()
            
                        
            # -------------------------------------------------------------------------------------------------------- #
            if rank == 0: 
                Model.graficar(cn.Vp, "Vp", corte, corte, corte)
                Model.graficar(cn.Vs, "Vs", corte, corte, corte)
                Model.graficar(cn.Rho, "rho", corte, corte, corte)
                Model.graficar(cn.Lambda, "\lambda", corte, corte, corte)
                Model.graficar(cn.Mu, "\mu", corte, corte, corte)
                plt.show()
            
        ctx.pop() #deactivate again
        ctx.detach() #delete it
            
        exit()
         
