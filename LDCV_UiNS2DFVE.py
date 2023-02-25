import math
import numpy as np
import matplotlib.pyplot as plt

#%% Constants
Re = 500.
St = 1.
U = 1.

#%% Domain
n = 51

L_x = 1
L_y = 1

x = np.linspace(0,L_x,n)
y = np.linspace(0,L_y,n)

dx = x[1]-x[0]
dy = y[1]-y[0]

xm = np.linspace(dx/2,L_x-dx/2,n-1) # midvalues
ym = np.linspace(dy/2,L_y-dy/2,n-1) # midvalues

xo = np.linspace(-dx/2,L_x+dx/2,n+1) # including ghost
yo = np.linspace(-dx/2,L_y+dy/2,n+1) # including ghost

#%% Time
t = 0
t_stop = 10.
CFL = 0.9
dt = dx/U

#%% Initialize

# u resides on x and ym
# v resides on xm and y
# p resides on xm and ym

u = np.zeros((n+2,n+1)) # including ghost
v = np.zeros((n+1,n+2)) # including ghost
p = np.zeros((n+2,n+2)) # including ghost

#%% Functions

def U_MOM_PREDICTOR(U, V, RE, ST, DX, DY, DT, IDX, JDX):
    
    # Velocities
    U_w = (U[JDX,IDX-1]+U[JDX,IDX])/2
    U_e = (U[JDX,IDX]+U[JDX,IDX+1])/2
    U_s = (U[JDX-1,IDX]+U[JDX,IDX])/2
    U_n = (U[JDX,IDX]+U[JDX+1,IDX])/2
    
    V_sw = V[JDX-1,IDX]
    V_se = V[JDX-1,IDX+1]
    V_s = (V_sw+V_se)/2
    V_nw = V[JDX,IDX]
    V_ne = V[JDX,IDX+1]
    V_n = (V_nw+V_ne)/2
    
    # Velocity derivatives 
    DUDX_w = (-U[JDX,IDX-1]+U[JDX,IDX])/DX
    DUDX_e = (-U[JDX,IDX]+U[JDX,IDX+1])/DX
    DUDY_s = (-U[JDX-1,IDX]+U[JDX,IDX])/DY
    DUDY_n = (-U[JDX,IDX]+U[JDX+1,IDX])/DY
        
    # # Pressures
    # P_w = P[JDX,IDX]
    # P_e = P[JDX,IDX+1]
    
    # Convective fluxes
    F_U_c_w = -U_w*U_w*DY
    F_U_c_e =  U_e*U_e*DY
    F_U_c_s = -U_s*V_s*DX
    F_U_c_n =  U_n*V_n*DX
    F_U_c = F_U_c_w + F_U_c_e + F_U_c_s + F_U_c_n
    
    # Diffusive fluxes
    F_U_d_w = -(1/RE)*DUDX_w*DY
    F_U_d_e =  (1/RE)*DUDX_e*DY 
    F_U_d_s = -(1/RE)*DUDY_s*DX
    F_U_d_n =  (1/RE)*DUDY_n*DX
    F_U_d = F_U_d_w + F_U_d_e + F_U_d_s + F_U_d_n
    
    # # Pressure fluxes
    # F_U_p_w = -P_w*DY
    # F_U_p_e =  P_e*DY
    # F_U_p = F_U_p_w + F_U_p_e
    
    # Unsteady term
    A_U = ST*(1/DT)*DX*DY
    
    # Update u
    # U_star = (-F_U_c + F_U_d - F_U_p)/A_U + U[JDX,IDX]
    U_star = (-F_U_c + F_U_d)/A_U + U[JDX,IDX]
    
    return U_star

def V_MOM_PREDICTOR(U, V, RE, ST, DX, DY, DT, IDX, JDX):
    
    # Velocities
    V_w = (V[JDX,IDX-1]+V[JDX,IDX])/2
    V_e = (V[JDX,IDX]+V[JDX,IDX+1])/2
    V_s = (V[JDX-1,IDX]+V[JDX,IDX])/2
    V_n = (V[JDX,IDX]+V[JDX+1,IDX])/2
    
    U_sw = U[JDX,IDX-1]
    U_nw = U[JDX+1,IDX-1]
    U_w = (U_sw+U_nw)/2
    U_se = U[JDX,IDX]
    U_ne = U[JDX+1,IDX]
    U_e = (U_se+U_ne)/2

    # Velocity derivatives
    DVDX_w = (-V[JDX,IDX-1]+V[JDX,IDX])/DX
    DVDX_e = (-V[JDX,IDX]+V[JDX,IDX+1])/DX
    DVDY_s = (-V[JDX-1,IDX]+V[JDX,IDX])/DY
    DVDY_n = (-V[JDX,IDX]+V[JDX+1,IDX])/DY
    
    # # Pressures
    # P_s = P[JDX,IDX]
    # P_n = P[JDX+1,IDX]
    
    # Convective fluxes
    F_V_c_w = -U_w*V_w*DY
    F_V_c_e =  U_e*V_e*DY
    F_V_c_s = -V_s*V_s*DX
    F_V_c_n =  V_n*V_n*DX
    F_V_c = F_V_c_w + F_V_c_e + F_V_c_s + F_V_c_n
    
    # Diffusive fluxes
    F_V_d_w = -(1/RE)*DVDX_w*DY
    F_V_d_e =  (1/RE)*DVDX_e*DY
    F_V_d_s = -(1/RE)*DVDY_s*DY
    F_V_d_n =  (1/RE)*DVDY_n*DY
    F_V_d = F_V_d_w + F_V_d_e + F_V_d_s + F_V_d_n
    
    # # Pressures
    # F_V_p_s = -P_s*DX
    # F_V_p_n =  P_n*DX    
    # F_V_p = F_V_p_s + F_V_p_n
    
    # Unsteady term
    A_V = ST*(1/DT)*DX*DY
    
    # Update u
    # V_star = (-F_V_c + F_V_d - F_V_p)/A_V + V[IDX,JDX]
    V_star = (-F_V_c + F_V_d)/A_V + V[IDX,JDX]
    
    return V_star

def apply_BCs(U, V, P, N, U_TOP):
    U[:,0] = 0
    U[:,N] = 0
    U[0,:] = -U[1,:]
    U[N+1,:] = 2*U_TOP - U[N,:]
    
    V[0,:] = 0
    V[N,:] = 0
    V[:,0] = -V[:,1]
    V[:,N+1] = -V[:,N]
    
    P[0,:] = P[1,:]
    P[N+1,:] = P[N,:]
    P[:,0] = P[:,1]
    P[:,N+1] = P[:,N]
    

#%% Integration

apply_BCs(u, v, p, n, U)

u_star = np.zeros_like(u)
v_star = np.zeros_like(v)
p_star = np.zeros_like(p)

for idx in range(1,n):
    for jdx in range(1,n+1):
        u_star[jdx,idx] = U_MOM_PREDICTOR(u, v, Re, St, dx, dy, dt, idx, jdx)
        
for idx in range(1,n+1):
    for jdx in range(1,n):
        v_star[jdx,idx] = V_MOM_PREDICTOR(u, v, Re, St, dx, dy, dt, idx, jdx)

apply_BCs(u_star, v_star, p, n, U)
