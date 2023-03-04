import math
import numpy as np
import matplotlib.pyplot as plt

#%% Constants
Re = 400.
St = 1.
u_lid = 1.

# Physical parameters
u_phys = 1.0 #[m/s]
rho_phys = 1000 #[kg/m3]
mu_phys = 0.00089 #[Pa-s]

L_phys = Re*mu_phys/u_phys
t_phys = L_phys/(u_phys*St)

#%% Domain
n = 32 # Even number
idx_m = int(n/2-1)

Length_x = 1
Length_y = 1

x = np.linspace(0,Length_x,n+1)
y = np.linspace(0,Length_y,n+1)

dx = x[1]-x[0]
dy = y[1]-y[0]

xm = np.linspace(dx/2,Length_x-dx/2,n) # midvalues
ym = np.linspace(dy/2,Length_y-dy/2,n) # midvalues

xo = np.linspace(-dx/2,Length_x+dx/2,n) # including ghost
yo = np.linspace(-dx/2,Length_y+dy/2,) # including ghost

xu,yu = np.meshgrid(x,ym)
xv,yv = np.meshgrid(xm,y)
xp,yp = np.meshgrid(xm,ym)

#%% Time
t = 0
t_stop = 10
CFL = 0.4
dt = CFL*dx/u_lid
n_t = int((t_stop-t)/dt + 1)

n_p_iter = 1

#%% Initialize

# u resides on x and ym
# v resides on xm and y
# p resides on xm and ym

u = np.zeros((n_t,n+2,n+1)) # including ghost nodes
v = np.zeros((n_t,n+1,n+2)) # including ghost nodes
p = np.zeros((n_t,n,n)) # no ghost nodes
div = np.zeros((n_t,n,n)) # no ghost nodes
u_mid = np.zeros((n_t,n,n)) # no ghost nodes
v_mid = np.zeros((n_t,n,n)) # no ghost nodes
vel_mid = np.zeros((n_t,n,n)) # no ghost nodes 
vort = np.zeros((n_t,n,n)) # no ghost nodes 
u_cl = np.zeros((n_t,n)) # no ghost nodes 
v_cl = np.zeros((n_t,n)) # no ghost nodes 
#%% Assemble Poisson solver

idxs_diag = np.diag_indices(n)
idxs_diag_p1 = (idxs_diag[0],idxs_diag[1]+1)
idxs_diag_m1 = (idxs_diag[0],idxs_diag[1]-1)

I_x = np.zeros((n,n))
I_x[idxs_diag] = 1

I_y = np.zeros((n,n))
I_y[idxs_diag] = 1

L_x = np.zeros((n,n))
L_x[idxs_diag_m1[0][1:n],idxs_diag_m1[1][1:n]] = -1
L_x[idxs_diag] = 2
L_x[idxs_diag_p1[0][0:n-1],idxs_diag_p1[1][0:n-1]] = -1
L_x[0,0] = 1
L_x[0,1] = -1
L_x[-1,-1] = 1
L_x[-1,-2] = -1


S_x = np.zeros((n,n))
S_y = np.zeros((n,n))
lambda_x = np.zeros((n,n))
lambda_y = np.zeros((n,n))
for mdx in range(0,n):
    for jdx in range(0,n):
        S_x[jdx,mdx]= np.cos(xm[jdx]*np.pi*mdx)
        S_y[jdx,mdx]= np.cos(ym[jdx]*np.pi*mdx)
        lambda_x[jdx,mdx] = (1/dx**2)*(2-2*np.cos(mdx*np.pi*dx))
        lambda_y[mdx,jdx] = (1/dy**2)*(2-2*np.cos(mdx*np.pi*dy))
lam = lambda_y + lambda_x
iS_x = np.linalg.inv(S_x)
iS_y = np.linalg.inv(S_y)


L_y = np.zeros((n,n))
L_y[idxs_diag_m1[0][1:n],idxs_diag_m1[1][1:n]] = -1
L_y[idxs_diag] = 2
L_y[idxs_diag_p1[0][0:n-1],idxs_diag_p1[1][0:n-1]] = -1
L_y[0,0] = 1
L_y[0,1] = -1
L_y[-1,-1] = 1
L_y[-1,-2] = -1

L = np.kron((1/dx**2)*L_x,I_y) + np.kron(I_x,(1/dy**2)*L_y)
# L[0,:] = 0.
# L[0,0] = 1.

D = 2/(dx**2) + 2/(dy**2)
Dx = D-(1/dx**2)
Dy = D-(1/dy**2)
Dxy = D-(1/dx**2)-(1/dy**2)


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
    
    # # Convective fluxes 
    # F_U_c_w = -U_w*U_w*DY
    # F_U_c_e =  U_e*U_e*DY
    # F_U_c_s = -U_s*V_s*DX
    # F_U_c_n =  U_n*V_n*DX
    # F_U_c = F_U_c_w + F_U_c_e + F_U_c_s + F_U_c_n
    
    # # Diffusive fluxes (FOR WHATEVER REASON, IT WORKS TO DO THE MOM. EQN. IN ONE STEP)
    # F_U_d_w = -(1/RE)*DUDX_w*DY
    # F_U_d_e =  (1/RE)*DUDX_e*DY 
    # F_U_d_s = -(1/RE)*DUDY_s*DX
    # F_U_d_n =  (1/RE)*DUDY_n*DX
    # F_U_d = F_U_d_w + F_U_d_e + F_U_d_s + F_U_d_n

    # Unsteady term
    A_U = ST*(1/DT)*DX*DY
    
    # Update u
    # U_star = (-F_U_c + F_U_d)/A_U + U[JDX,IDX]
    U_star = U[JDX,IDX] + (1/A_U) * ( -(U_e*U_e*DY - U_w*U_w*DY + U_n*V_n*DX - U_s*V_s*DX) + (1/RE)*(-DUDX_w*DY + DUDX_e*DY -DUDY_s*DX + DUDY_n*DX) )
    
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
    
    # # Convective fluxes
    # F_V_c_w = -U_w*V_w*DY
    # F_V_c_e =  U_e*V_e*DY
    # F_V_c_s = -V_s*V_s*DX
    # F_V_c_n =  V_n*V_n*DX
    # F_V_c = F_V_c_w + F_V_c_e + F_V_c_s + F_V_c_n
    
    # # Diffusive fluxes (FOR WHATEVER REASON, IT WORKS TO DO THE MOM. EQN. IN ONE STEP)
    # F_V_d_w = -(1/RE)*DVDX_w*DY
    # F_V_d_e =  (1/RE)*DVDX_e*DY
    # F_V_d_s = -(1/RE)*DVDY_s*DX
    # F_V_d_n =  (1/RE)*DVDY_n*DX
    # F_V_d = F_V_d_w + F_V_d_e + F_V_d_s + F_V_d_n

    # Unsteady term
    A_V = ST*(1/DT)*DX*DY
    
    # Update v
    # V_star = (-F_V_c + F_V_d)/A_V + V[IDX,JDX]
    V_star = V[JDX,IDX] + (1/A_V) * ( -(-V_s*V_s*DX + V_n*V_n*DX - U_w*V_w*DX + U_e*V_e*DX) + (1/RE)*(-DVDX_w*DY + DVDX_e*DY -DVDY_s*DX + DVDY_n*DX) )
    
    return V_star

def U_MOM_CORRECTOR(U, P, ST, DX, DY, DT, IDX, JDX):
    
    # Pressures
    P_w = P[JDX-1,IDX-1]
    P_e = P[JDX-1,IDX]
    
    # Pressure fluxes
    F_U_p_w = -P_w*DY
    F_U_p_e =  P_e*DY
    F_U_p = F_U_p_w + F_U_p_e
    
    # Unsteady term
    A_U = ST*(1/DT)*DX*DY
    
    # Update u
    U_next = (-F_U_p)/A_U + U[JDX,IDX]
    
    return U_next

def V_MOM_CORRECTOR(V, P, ST, DX, DY, DT, IDX, JDX):
    
    # Pressures
    P_s = P[JDX-1,IDX-1]
    P_n = P[JDX,IDX-1]
    
    # Pressures
    F_V_p_s = -P_s*DX
    F_V_p_n =  P_n*DX    
    F_V_p = F_V_p_s + F_V_p_n
    
    # Unsteady term
    A_U = ST*(1/DT)*DX*DY
    
    # Update u
    V_next = (-F_V_p)/A_U + V[JDX,IDX]
    
    return V_next

def apply_velocity_BCs(U, V, N, U_TOP):
    U[:,0] = 0
    U[:,N] = 0
    U[0,:] = -U[1,:]
    U[N+1,:] = 2*U_TOP - U[N,:]
    
    V[0,:] = 0
    V[N,:] = 0
    V[:,0] = -V[:,1]
    V[:,N+1] = -V[:,N]
    
def step_time(U, V, RE, ST, DX, DY, DT, N, L_MAT, U_LID):
    
    apply_velocity_BCs(U, V, N, U_LID)
    
    # Predictor step
    U_STAR = np.zeros_like(U)
    V_STAR = np.zeros_like(V)
    
    for IDX in range(1,N):
        for JDX in range(1,N+1):
            U_STAR[JDX,IDX] = U_MOM_PREDICTOR(U, V, RE, ST, DX, DY, DT, IDX, JDX)
    for IDX in range(1,N+1):
        for JDX in range(1,N):   
            V_STAR[JDX,IDX] = V_MOM_PREDICTOR(U, V, RE, ST, DX, DY, DT, IDX, JDX)
  
    
    # Assemble right-hand side of pressure Poisson equation
    R = np.zeros((N,N))
    for IDX in range(0,N):
        for JDX in range(0,N):
            R[JDX,IDX] = -ST*(1/DT)*((U_STAR[JDX+1,IDX+1]-U_STAR[JDX+1,IDX])/DX + (V_STAR[JDX+1,IDX+1]-V_STAR[JDX,IDX+1])/DY)
    
    # Solve pressure Poisson equation
    # P_VEC = np.linalg.solve(L_MAT,np.matrix.flatten(R,order="C"))
    # P = np.reshape(P_VEC,(N,N),order="C")
    SOL = np.matmul(iS_y,np.matmul(R,np.transpose(iS_x)))
    SOL = SOL/lam
    SOL[0,0] = 0
    SOL = np.matmul(S_y,np.matmul(SOL,np.transpose(S_x)))
    P = np.copy(SOL)
    
    # Corrector step
    U_NEXT = np.zeros_like(U)
    V_NEXT = np.zeros_like(V)
    
    for IDX in range(1,N):
        for JDX in range(1,N+1):
            U_NEXT[JDX,IDX] = U_MOM_CORRECTOR(U_STAR, P, ST, DX, DY, DT, IDX, JDX)
                
    for IDX in range(1,N+1):
        for JDX in range(1,N):
            V_NEXT[JDX,IDX] = V_MOM_CORRECTOR(V_STAR, P, ST, DX, DY, DT, IDX, JDX)

    
    # Compute divergence of flowfield and interpolate velocities onto pressure grid
    DIV = np.zeros((N,N))
    U_MID = np.zeros((N,N))
    V_MID = np.zeros((N,N))
    VEL_MID = np.zeros((N,N))
    VORT = np.zeros((N,N))
    
    for IDX in range(0,N):
        for JDX in range(0,N):
            DIV[JDX,IDX] = ((U_NEXT[JDX+1,IDX+1]-U_NEXT[JDX+1,IDX])/DX + (V_NEXT[JDX+1,IDX+1]-V_NEXT[JDX,IDX+1])/DY)
            U_MID[JDX,IDX] = (U_NEXT[JDX+1,IDX+1]+U_NEXT[JDX+1,IDX])/2
            V_MID[JDX,IDX] = (V_NEXT[JDX+1,IDX+1]+V_NEXT[JDX,IDX+1])/2
            VEL_MID[JDX,IDX] = np.sqrt((U_MID[JDX,IDX]**2)+(V_MID[JDX,IDX]**2))       
            VORT[JDX,IDX] = ((V_NEXT[JDX+1,IDX+1] - V_NEXT[JDX,IDX+1])/DX - (U_NEXT[JDX+1,IDX+1] - U_NEXT[JDX+1,IDX])/DY)
    
    
    U_MID_1 = U_MID[:,idx_m]
    U_MID_2 = U_MID[:,idx_m+1]
    U_CL = (U_MID_1 + U_MID_2)/2
    
    V_MID_1 = V_MID[idx_m,:]
    V_MID_2 = V_MID[idx_m+1,:]
    V_CL = (V_MID_1 + V_MID_2)/2
    
    return U_NEXT, V_NEXT, P, DIV, U_MID, V_MID, VEL_MID, VORT, U_CL, V_CL
    

#%% Integration
tdx = 0

while tdx < n_t-1:
    print("t = "+"{:.4f}".format(t))
    
    u_n = np.copy(u[tdx,:,:])
    v_n = np.copy(v[tdx,:,:])
    
    # RK4 Integration
    u_k1, v_k1, p_k1, div_k1, u_mid_k1, v_mid_k1, vel_mid_k1, vort_k1, u_cl_k1, v_cl_k1  =    step_time(u_n, v_n, Re, St, dx, dy, dt, n, L, u_lid)
    u_k2, v_k2, p_k2, div_k2, u_mid_k2, v_mid_k2, vel_mid_k2, vort_k2, u_cl_k2, v_cl_k2  =    step_time(u_n+0.5*dt*u_k1, v_n+0.5*dt*v_k1, Re, St, dx, dy, dt, n, L, u_lid)
    u_k3, v_k3, p_k3, div_k3, u_mid_k3, v_mid_k3, vel_mid_k3, vort_k3, u_cl_k3, v_cl_k3  =    step_time(u_n+0.5*dt*u_k2, v_n+0.5*dt*v_k2, Re, St, dx, dy, dt, n, L, u_lid)
    u_k4, v_k4, p_k4, div_k4, u_mid_k4, v_mid_k4, vel_mid_k4, vort_k4, u_cl_k4, v_cl_k4  =    step_time(u_n+1.0*dt*u_k3, v_n+1.0*dt*v_k3, Re, St, dx, dy, dt, n, L, u_lid)
    
    # RK4 terms
    # u_np1 = u_n + (1/6)*(u_k1 + 2*u_k2 + 2*u_k3 + u_k4)*dt
    # v_np1 = v_n + (1/6)*(v_k1 + 2*v_k2 + 2*v_k3 + v_k4)*dt
    
    # Explicit Euler terms (Upgrade such that the functions which obtain these parameters are outside of time integration)
    u_np1 = u_k1
    v_np1 = v_k1
    p_np1 = p_k1
    div_np1 = div_k1
    u_mid_np1 = u_mid_k1
    v_mid_np1 = v_mid_k1
    vel_mid_np1 = vel_mid_k1
    vort_np1 = vort_k1
    u_cl_np1 = u_cl_k1
    v_cl_np1 = v_cl_k1
        
    u[tdx+1,:,:]=np.copy(u_np1) 
    v[tdx+1,:,:]=np.copy(v_np1)
    p[tdx+1,:,:]=np.copy(p_np1)
    div[tdx+1,:,:]=np.copy(div_np1)
    u_mid[tdx+1,:,:]=np.copy(u_mid_np1)
    v_mid[tdx+1,:,:]=np.copy(v_mid_np1)
    vel_mid[tdx+1,:,:]=np.copy(vel_mid_np1)
    vort[tdx+1,:,:]=np.copy(vort_np1)
    u_cl[tdx+1,:]=np.copy(u_cl_np1)
    v_cl[tdx+1,:]=np.copy(v_cl_np1)
    
    
    t = t + dt
    tdx = tdx + 1
    
#%% Plotting
plt.close('all')

cmap_choose = 'jet'

plt.figure()
plt.subplot(2,4,1)
plt.contourf(xu,yu,u[-1,1:-1,:],cmap = cmap_choose,levels = 5*n, origin='lower')
plt.colorbar()
plt.title("$u$")
plt.xlabel("$x$")
plt.ylabel("$y$")

plt.subplot(2,4,2)
plt.contourf(xv,yv,v[-1,:,1:-1],cmap = cmap_choose,levels = 5*n, origin='lower')
plt.colorbar()
plt.title("$v$")
plt.xlabel("$x$")
plt.ylabel("$y$")

plt.subplot(2,4,3)
plt.contourf(xp,yp,vel_mid[-1,:,:],cmap = cmap_choose ,levels = 5*n, origin='lower')
plt.colorbar()
plt.streamplot(xp,yp,u_mid[-1,:,:],v_mid[-1,:,:],color="white",density = 4,linewidth=0.5)
plt.xlim([xm[0],xm[-1]])
plt.ylim([ym[0],ym[-1]])
plt.title("$|u_{i}|$")
plt.xlabel("$x$")
plt.ylabel("$y$")

plt.subplot(2,4,4)
plt.plot(u_cl[-1,:],ym,'k')
plt.title("$x/L = 0.5$")
plt.xlabel("$u$")
plt.ylabel("$y$")
plt.grid()

plt.subplot(2,4,5)
plt.contourf(xp,yp,p[-1,:,:],cmap = cmap_choose,levels = 5*n, origin='lower')
plt.colorbar()
plt.streamplot(xp,yp,u_mid[-1,:,:],v_mid[-1,:,:],color="white",density = 4,linewidth=0.5)
plt.title(r"$p$")
plt.xlim([xm[0],xm[-1]])
plt.ylim([ym[0],ym[-1]])
plt.xlabel("$x$")
plt.ylabel("$y$")

plt.subplot(2,4,6)
plt.contourf(xp,yp,div[-1,:,:],cmap = cmap_choose,levels = 5*n, origin='lower')
plt.colorbar()
plt.title(r"$\frac{\partial{u_{i}}}{\partial{x_{i}}}$")
plt.xlabel("$x$")
plt.ylabel("$y$")

plt.subplot(2,4,7)
plt.contourf(xp,yp,vel_mid[-1,:,:],cmap = cmap_choose,levels = 5*n, origin='lower')
plt.colorbar()
plt.xlim([xm[0],xm[-1]])
plt.ylim([ym[0],ym[-1]])
plt.title("$|u_{i}|$")
plt.xlabel("$x/L$")
plt.ylabel("$y/L$")

plt.subplot(2,4,8)
plt.plot(xm,v_cl[-1,:],'k')
plt.title("$y/L = 0.5$")
plt.xlabel("$x$")
plt.ylabel("$v$")
plt.grid()

plt.suptitle("t = "+"{:.4f}".format(t)+", Re = "+"{:.0f}".format(Re)+", St = "+"{:.0f}".format(St)+", N = "+"{:.0f}".format(n))
# plt.tight_layout()