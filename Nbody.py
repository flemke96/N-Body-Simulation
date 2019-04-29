# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 22:17:56 2019

@author: Florian
"""

import numpy as np
from numpy.random import normal
from scipy import constants
import matplotlib.pyplot as plt


#%% 

G = constants.G     # gravitational constant

mode = 3

# =================================================================================
# ========= MODE 1 ==========
# small mass large mass only in x-y-plane with intial velocity of small mass

if mode == 1:
    N = 2                     # number of bodies
    Time = 1e+3               # length of time to be calculated
    dt = 1e-1                # length of one timestep in seconds
    timesteps = int(Time/dt)  # discretization of time
    EPS = 1e-4                # softening parameter to avoid singularities when r2-r1=0

    r = np.zeros([N,timesteps,3])   # position array
    v = np.zeros([N,timesteps,3])   # velocity array
    f = np.zeros([N,timesteps,3])   # force array
    a = np.zeros([N,timesteps,3])   # acceleration array
    
    m = np.ones(N)                  # mass array
    
    # ==================
    # Initial conditions    
    # ==================
    m[0] = 1e+6
    m[1] = 5e+5
    x0 = 1e+0
    v0 = 1.0e-2
    
    r[0,0,:] = 0    # rx, ry, rz big body
    r[1,0,0] = x0    # rx small body; y,z = 0
    
    v[0,0,:] = 0
    v[1,0,0] = 0
    v[1,0,1] = v0
    v[1,0,2] = 0


# =================================================================================
# ========= MODE 2 ==========
# three bodies of same mass with randomly assigned intitial velocities in x-y-plane

elif mode == 2:
    N = 3                     # number of bodies
    Time = 1e+3               # length of time to be calculated
    dt = 1e-2                # length of one timestep in seconds
    timesteps = int(Time/dt)  # discretization of time
    EPS = 1e-3                # softening parameter to avoid singularities when r2-r1=0

    r = np.zeros([N,timesteps,3])   # position array
    v = np.zeros([N,timesteps,3])   # velocity array
    f = np.zeros([N,timesteps,3])   # force array
    a = np.zeros([N,timesteps,3])   # acceleration array
    
    m = np.ones(N)                  # mass array

    # ==================
    # Initial conditions    
    # ==================
    size = .3e-0            # initial size of the box that frames the position
    v0_max = 1e-3
    m_all = 1e+3
    m = m_all*m
    
    for body_i in range(len(r)):
        r[body_i][0][0] = np.random.uniform(-size/2,size/2)
        r[body_i][0][1] = np.random.uniform(-size/2,size/2)
        r[body_i][0][2] = 0
        v[body_i][0][0] = np.random.uniform(-v0_max/2,v0_max/2)
        v[body_i][0][1] = np.random.uniform(-v0_max/2,v0_max/2)
        v[body_i][0][2] = 0
        

# =================================================================================
# ========= MODE 3 ==========
# a random example intitial conditions set from mode 2 with N=3
# just for experimenting with visualization, should not be changed
        
elif mode == 3:
    N = 3                     # number of bodies
    Time = .5e+3               # length of time to be calculated
    dt = 1e-2                # length of one timestep in seconds
    timesteps = int(Time/dt)  # discretization of time
    EPS = 1e-3                # softening parameter to avoid singularities when r2-r1=0

    r = np.zeros([N,timesteps,3])   # position array
    v = np.zeros([N,timesteps,3])   # velocity array
    f = np.zeros([N,timesteps,3])   # force array
    a = np.zeros([N,timesteps,3])   # acceleration array
    
    m = np.ones(N)                  # mass array

    # ==================
    # Initial conditions    
    # ==================
    size = .3e-0            # initial size of the box that frames the position
    v0_max = 1e-3
    m_all = 1e+3
    m = m_all*m
    
# BODY 0
    r[0][0][0] = 0.04560677906771746
    r[0][0][1] = -0.010447214141804995
    r[0][0][2] = 0.0
    v[0][0][0] = 0.00021037498322122323
    v[0][0][1] = 0.00042589630891071606
    v[0][0][2] = 0.0
# BODY 1
    r[1][0][0] = -0.06430803939366223
    r[1][0][1] = -0.04961162041220647
    r[1][0][2] = 0.0
    v[1][0][0] = 4.330072837055497e-05
    v[1][0][1] = 4.980257351787148e-05
    v[1][0][2] = 0.0
# BODY 2
    r[2][0][0] = -0.007974020697223638
    r[2][0][1] = -0.12191953154496123
    r[2][0][2] = 0.0
    v[2][0][0] = 0.00033100843164763207
    v[2][0][1] = 0.0003563859229114122
    v[2][0][2] = 0.0


#%%

print(chr(27) + "[2J")    

print('Time =',Time,'s')
print('dt =',dt,'s')
print('total steps =', timesteps)


def grav_force(m1,m2,r1,r2):
    return (G*m1*m2) * (r2-r1) / (np.linalg.norm(r2-r1)**3 + EPS**2)

#def rad_force(dist,v,m):
#    return m*v**2/dist



#%% Calculation of position for all times


for t_i in range(timesteps):
    if t_i%1e+4==0:
        print('current timestep =', t_i)
    for body_i in range(len(r)):
        for body_k in range(len(r)):
            if body_k != body_i:
                tmp = grav_force(m[body_i],m[body_k],r[body_i][t_i],r[body_k][t_i])
                f[body_i][t_i] += tmp
        a[body_i][t_i] = f[body_i][t_i] / m[body_i]
        if t_i < (timesteps-1):
            v[body_i][t_i+1] = v[body_i][t_i] + a[body_i][t_i] * dt
            r[body_i][t_i+1] = r[body_i][t_i] + v[body_i][t_i] * dt + 1/2 * a[body_i][t_i] * dt**2



#%% VISUALISATION


X = np.zeros([N,timesteps])
Y = np.zeros([N,timesteps])
Z = np.zeros([N,timesteps])


colorlist = ['red','blue', 'green', 'cyan', 'orange', 'brown']

plt.figure(1)


for body_i in range(N):
    X[body_i] = r[body_i,:,0]
    Y[body_i] = r[body_i,:,1]
    Z[body_i] = r[body_i,:,2]
    plt.subplot(1,1,1)
    plt.plot(X[body_i],                 # plot path of bodies
             Y[body_i],
             color=colorlist[body_i])
    plt.plot(X[body_i,timesteps-1],     # plot last calculated position of bodies
             Y[body_i,timesteps-1],
             color=colorlist[body_i],
             marker='o')


plt.xlabel('$r_x$ in m')
plt.ylabel('$r_y$ in m')
#plt.legend([line1,
#            line2,
#            line3,
#            line4], loc = 'lower right')

plt.tight_layout()
plt.show()
#plt.savefig('3bodies_random.pdf',bboxinches='tight')


print('Computation completed.\nShow animation (y/n)?')
answer = input()

#%%


xmax = 0.2
xmin = -0.1
ymax = 0.15
ymin = -0.15


if answer == 'y':
    dt_plot = 500

    plt.figure(2)
    
    for t_i in range(0,timesteps,dt_plot):
        for body_i in range(N):
            plt.subplot(1,1,1)
            plt.plot(X[body_i,0:t_i],
                     Y[body_i,0:t_i],
                     color=colorlist[body_i])
            plt.plot(X[body_i,t_i],
                     Y[body_i,t_i],
                     color=colorlist[body_i],
                     marker='o')
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax)
        print(chr(27) + "[2J")
        plt.show()

#%%



