# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:14:21 2019

@author: Florian
"""

import numpy as np
from numpy.random import normal
from numpy.linalg import norm
from scipy import constants
from matplotlib import pyplot as plt
from matplotlib import animation

#%%

G = constants.G
EPS = 1e-3

class Body:
    def __init__(self,m,r0,v0):
        self.m = m
        self.r0 = r0
        self.v0 = v0
        
#        self.acceleration = [ax,ay,az]
#        self.velocity = [vx,vy,vz]
#        self.position = [rx,rv,rz]
        
    def dist(self,other):
        return np.abs(other.r - self.r)
    
    def grav_accel(self,other):
        d = Body.dist(self.other)
        return G * other.m / (d**2 + EPS**2) * (other.r-self.r) / d

# ========== INTITIALIZING 3 BODIES ===========
m = 1e+3
m1 = m
m2 = m
m3 = m

r0_1 = [0.04560677906771746,-0.010447214141804995,0.0]
r0_2 = [-0.06430803939366223,-0.04961162041220647,0.0]
r0_3 = [-0.007974020697223638,-0.12191953154496123,0.0]

v0_1 = [0.00021037498322122323,0.00042589630891071606,0.0]
v0_2 = [4.330072837055497e-05,4.980257351787148e-05,0.0]
v0_3 = [0.00033100843164763207,0.0003563859229114122,0.0]

body1 = Body(m1,r0_1,v0_1)
body2 = Body(m2,r0_2,v0_2)
body3 = Body(m3,r0_3,v0_3)

bodies = (body1,body2,body3)

#%% ========== COMPUTATION ============

# x- and y-axes boundaries
xmax = 0.2
xmin = -0.1
ymax = 0.15
ymin = -0.15

# defining X- and Y-values for the 3 different lines for the plot animation
# for plotting the position (x over y)
for body_i in range(N):
    X[body_i] = r[body_i,:,0]
    Y[body_i] = r[body_i,:,1]
    Z[body_i] = r[body_i,:,2]



# actual animation part
fig = plt.figure()
ax = plt.axes(xlim=(xmin,xmax),
              ylim=(ymin,ymax))
line1, line2, line3 = ax.plot([], [],
                              [], [],
                              [], [], lw=1)

def init():
    line1.set_data([],[])
    line2.set_data([],[])
    line3.set_data([],[])
    return line1,line2,line3


# animation function
def animate(t_i):
    
    
    line1.set_data(X[0,0:t_i],Y[0,0:t_i])
    line2.set_data(X[1,0:t_i],Y[1,0:t_i])
    line3.set_data(X[2,0:t_i],Y[2,0:t_i])

    return line1, line2, line3


#dt_plot = 500
dt_plot = 200

anim = animation.FuncAnimation(fig, func=animate, init_func=init,
                               frames=range(0,timesteps,dt_plot),
                               interval=30, blit=True,
                               repeat=True, repeat_delay=500)

#anim.save('basic_animation.gif', fps=30, extra_args=['-vcodec', 'libx264'])
#anim.save('basic_animation.gif')

plt.show


