'''
PLOT the optimal trajectory of states and inputs
of the AWE system in drag mode
Python Version 2.7 / Casadi version 2.4.1
- Author: Elena Malz, Chalmers 2016
'''
import json
import time

import casadi as ca
import casadi.tools as ca
import numpy as np
from numpy import pi
import pickle
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import scipy as sp
from matplotlib.ticker import ScalarFormatter
from values_dragonfly_small import initial_params

params = initial_params()




plt.close('all')
ScalePower = 1e-3

with open('solution.dat', 'r') as f:
    (val_opt, opt, nk, d) = pickle.load(f)

with open('init.dat', 'r') as f:
    (val_init, vars_init, nk, d) = pickle.load(f)

with open('cost.dat', 'r') as f:
    (E_final, Lifting, Tracking, Cost, Reg)= pickle.load(f)


tau_root = ca.collocationPoints(d,'radau')

Tx = np.zeros((nk,d+1))
for k in range(nk):
    for j in range(d+1):
        Tx[k,j] = (k + tau_root[j])*opt['tf']/float(nk)

Tu = np.zeros((nk))
for k in range(nk):
        Tu[k] = k *opt['tf']/float(nk)

Txa = np.zeros((nk,d))
for k in range(nk):
    for j in range(d):
        Txa[k,j] = (k + tau_root[j+1])*opt['tf']/float(nk)

tgrid_x  = Tx.reshape(nk*(d+1))
tgrid_u  = Tu.reshape(nk)
tgrid_xa = Txa.reshape(nk*(d))



# --- COST FUNCION ---

plt.ion()
plt.figure()
ax = plt.subplot(111)
track0 = ax.bar(1,np.array(Tracking),0.1, color = 'r')
regu0 = ax.bar(1+0.1,np.array(Reg),0.1, color = 'b')
lifting0 = ax.bar(1+0.2,np.array(Lifting),0.1, color = 'k')
energy0 = ax.bar(1+0.3,np.array(E_final*-1),0.1, color = 'g')
cost0 = ax.bar(1+0.4,np.array(Cost),0.1, color = 'y')
ax.legend((track0,regu0,lifting0,energy0,cost0), (('tracking','regularisation','lifting','energy*-1','total cost')))
plt.grid('on')
#ax.set_ylim([0,3000])
plt.show()


plt.ion()
plt.figure()
plt.plot(tgrid_xa,val_opt['speed'])
plt.title('speed')
plt.grid('on')


plt.figure()
k = 1
for name in set(['AoA_deg','sslip_deg']):
    plt.subplot(2,3,k)
    plt.title(name)
    plt.plot(tgrid_xa,val_opt[name])
    plt.plot(tgrid_xa,val_init[name])
    plt.grid('on')
    k+=1

for name in set(['CD', 'CL']):
    plt.subplot(2,3,k+1)
    plt.title(name)
    plt.plot(tgrid_xa,val_opt[name])
    plt.plot(tgrid_xa,val_init[name])
    plt.grid('on')
    k+=1
plt.subplot(2,3,6)
plt.plot(tgrid_xa, [L/D for L,D in zip(val_opt['CL'],val_opt['CD'])])
plt.grid('on')
plt.title('L/D')

# POSITION , SPEED, ANGULAR VEL
plt.figure()
plt.clf()
for k in range(3):
    plt.subplot(3,3,k+1)
    plt.plot(tgrid_x,opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',k].T)
    plt.plot(tgrid_x,vars_init["Xd",ca.horzcat,:,ca.horzcat,:,'q',k].T)
    plt.title('q'+str(k))
    plt.grid('on')
    plt.subplot(3,3,k+4)
    plt.plot(tgrid_x,opt["Xd",ca.horzcat,:,ca.horzcat,:,'dq',k].T)
    plt.plot(tgrid_x,vars_init["Xd",ca.horzcat,:,ca.horzcat,:,'dq',k].T)
    plt.title('dq'+str(k))
    plt.grid('on')
    plt.subplot(3,3,k+7)
    plt.plot(tgrid_x,opt["Xd",ca.horzcat,:,ca.horzcat,:,'w',k].T)
    plt.plot(tgrid_x,vars_init["Xd",ca.horzcat,:,ca.horzcat,:,'w',k].T)
    plt.title('w'+str(k))
    plt.grid('on')

# FICTIOUS AND REAL CONTROLS
plt.figure()
for k in range(3):
    plt.subplot(3,3,k+1)
    plt.step(tgrid_u,opt['U',ca.horzcat,:,'u', k].T)
    plt.step(tgrid_u,vars_init['U',ca.horzcat,:,'u', k].T)
    plt.title('homotopy_accels_v'+str(k))
    plt.grid('on')

    plt.subplot(3,3,k+4)
    plt.step(tgrid_u,opt['U',ca.horzcat,:,'T', k].T)
    plt.step(tgrid_u,vars_init['U',ca.horzcat,:,'T', k].T)
    plt.title('homotopy_accels_omega'+str(k))
    plt.grid('on')

    plt.subplot(3,3,k+7)
    plt.step(tgrid_x,opt['Xd',ca.horzcat,:,ca.horzcat,:,'coeff',k].T*np.ones(tgrid_x.shape[0])*180/pi)
    plt.step(tgrid_x,vars_init['Xd',ca.horzcat,:,ca.horzcat,:,'coeff',k].T*np.ones(tgrid_x.shape[0])*180/pi)
    plt.title('coeff'+str(k))
    plt.grid('on')


# # FORCES
# plt.figure()
# for k in range(3):
#     plt.subplot(4,1,k+1)
#     plt.plot(tgrid_xa, [o[k] for o in val_opt['F_aero'] ])
#     plt.plot(tgrid_xa, [o[k] for o in val_opt['F_tether'] ])
#     plt.plot(tgrid_xa, [o[k] for o in val_opt['F_drag'] ])
#     plt.grid('on')
#     plt.legend(['F_aero', 'F_tether','F_drag'])
# plt.subplot(4,1,4)
# plt.plot(tgrid_xa, val_opt['power'])
# plt.title('OUTPUT - POWER')
# plt.grid('on')

# FORCES
plt.figure()
plt.subplot(4,1,1)
plt.plot(tgrid_xa, [np.linalg.norm(val_opt['F_aero'][k]) for k in range(0,nk*d) ])
plt.grid('on'), plt.title('F_aero')
plt.subplot(4,1,2)
plt.plot(tgrid_xa, [np.linalg.norm(val_opt['F_tether'][k]) for k in range(0,nk*d) ])
plt.grid('on'), plt.title('F_tether')
ax = plt.subplot(4,1,3)
ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.plot(tgrid_xa, [np.linalg.norm(val_opt['F_drag'][k]) for k in range(0,nk*d) ])
plt.plot(tgrid_xa, [np.linalg.norm(val_opt['Tether_drag'][k]) for k in range(0,nk*d) ])
plt.grid('on'), plt.title(['F_drag', 'Tether_drag'])

plt.subplot(4,1,4)
plt.plot(tgrid_xa, val_opt['power'])
plt.title('OUTPUT - POWER')
plt.grid('on')


# ____ 3D PLOT /  TRAJECTORIES
fig = plt.figure()
ax = fig.add_subplot(2,2,1,projection='3d')
ax.plot(np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',0])[0],np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',1])[0],np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',2])[0],'-')
ax.plot(np.array(vars_init["Xd",ca.horzcat,:,ca.horzcat,:,'q',0])[0],np.array(vars_init["Xd",ca.horzcat,:,ca.horzcat,:,'q',1])[0],np.array(vars_init["Xd",ca.horzcat,:,ca.horzcat,:,'q',2])[0],'-')
ax.scatter(0,0,0,'r')
ax.plot([0, opt["Xd",0,0,'q',0]],[0, opt["Xd",0,0,'q',1]],[0,opt["Xd",0,0,'q',2]])
ax.scatter([opt["Xd",3,0,'q',0]],[opt["Xd",3,0,'q',1]],[opt["Xd",3,0,'q',2]],'g')
#ax.set_zlim3d(0.8, 1.1)
plt.axis('equal')

plt.subplot(2,2,2)
plt.plot(np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',0])[0],np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',1])[0],'-')
plt.plot(np.array(vars_init["Xd",ca.horzcat,:,ca.horzcat,:,'q',0])[0],np.array(vars_init["Xd",ca.horzcat,:,ca.horzcat,:,'q',1])[0],'-')
plt.title('x-y')
plt.grid('on')

plt.subplot(2,2,3)
plt.plot(np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',1])[0],np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',2])[0])
plt.plot(np.array(vars_init["Xd",ca.horzcat,:,ca.horzcat,:,'q',1])[0],np.array(vars_init["Xd",ca.horzcat,:,ca.horzcat,:,'q',2])[0],'-')
plt.title('y-z')
plt.grid('on')

plt.subplot(2,2,4)
plt.plot(np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',0])[0],np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',2])[0],'-')
plt.plot(np.array(vars_init["Xd",ca.horzcat,:,ca.horzcat,:,'q',0])[0],np.array(vars_init["Xd",ca.horzcat,:,ca.horzcat,:,'q',2])[0],'-')
plt.title('x-z')

plt.axis('equal')
plt.grid('on')
plt.show()


# --- LoD FIGURE DEPENDENT ON ANGLE OF ATTACK
plt.figure()
L   = []
D   = []
LoD = []
sslip = 0.0
alpha_list = np.arange(0,22*pi/180,0.01)
for AoA in alpha_list:
    L.append(0.8   * AoA/ (10*pi/180))
    D.append(0.008 + 0.04*(AoA/(10*pi/180))**2 + 2.*sslip**2)
    LoD.append((0.8   * AoA/ (10*pi/180))/(0.008 + 0.04*(AoA/(10*pi/180))**2 + 2.*sslip**2))

ax = plt.subplot(1,3,1)
ax.get_yaxis().get_major_formatter().set_useOffset(False)

plt.subplot(1,3,1); plt.plot(alpha_list*180/pi,L)
plt.grid('on')
plt.subplot(1,3,2); plt.plot(alpha_list*180/pi,D)
plt.grid('on')
plt.title('L,D,LoD')
plt.subplot(1,3,3); plt.plot(alpha_list*180/pi,LoD)
plt.grid('on')

# ---- WINDSHEAR
plt.figure()
plt.plot(tgrid_xa,val_opt['windspeed_shear'])
plt.grid('on')
plt.title('wind speed')


# --------------------------------------------------------
# Paper plots
# --------------------------------------------------------
axisfont = {'size':'28'}

fig = plt.figure()
plt.subplot(1,2,1)
plt.plot(np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',0])[0],np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',1])[0],'-')
plt.xlabel('x', **axisfont)
plt.ylabel('y', **axisfont)
plt.locator_params(nbins=3)
plt.tick_params(labelsize=25)
plt.grid('on')

plt.subplot(1,2,2)
plt.plot(np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',0])[0],np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',2])[0],'-')
plt.ylabel('z', **axisfont)
plt.xlabel('x', **axisfont)
plt.locator_params(nbins=3)
plt.grid('on')
plt.tick_params(labelsize=25)










# ## --- ROTATION MATRIX
# plt.figure()
# plt.clf()
# for i in range(3):
#     for j in range(3):
#         plt.subplot(3,3,j+3*i+1)
# #        print j+3*i+1
#         plt.plot(tgrid_x,opt["Xd",ca.horzcat,:,ca.horzcat,:,"R",i+3*j].T)
#         plt.plot(tgrid_x,vars_init["Xd",ca.horzcat,:,ca.horzcat,:,"R",i+3*j].T)
#
#         plt.title('R'+str(i+1)+str(j+1))
#         plt.grid('on')


# ##--- KITE FLYING

fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')

# normFaero = []
# for k in np.arange(0,nk,nk/10):
#     normFaero.append(np.linalg.norm(Faero_E[k]))
# max_normFaero = max(normFaero)

for k in np.arange(0,nk,nk/10):

#    b = 0.1*A
#    a = 0.3*A
    A = 20
    b = 0.5*0.15*A
    a = 0.5*0.85*A

    # ax.plot([float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0]+a*opt["Xd",k,0,"R",0])],
    #         [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1]+a*opt["Xd",k,0,"R",1])],
    #         [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2]+a*opt["Xd",k,0,"R",2])],color='red')
    #
    #
    # ax.plot([float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0]+a*opt["Xd",k,0,"R",3])],
    #         [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1]+a*opt["Xd",k,0,"R",4])],
    #         [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2]+a*opt["Xd",k,0,"R",5])],color='blue')
    #
    #
    # ax.plot([float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0]+a*opt["Xd",k,0,"R",6])],
    #         [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1]+a*opt["Xd",k,0,"R",7])],
    #         [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2]+a*opt["Xd",k,0,"R",8])],color='green')

    ax.plot(np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',0])[0],
            np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',1])[0],
            np.array(opt["Xd",ca.horzcat,:,ca.horzcat,:,'q',2])[0],linestyle = '--', color = 'grey',)


    vtx = np.array([
                       [float(opt["Xd",k,0,"q",0]+b*opt["Xd",k,0,"R",0]+a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]+b*opt["Xd",k,0,"R",1]+a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]+b*opt["Xd",k,0,"R",2]+a*opt["Xd",k,0,"R",5]) ],
                       [float(opt["Xd",k,0,"q",0]-b*opt["Xd",k,0,"R",0]+a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]-b*opt["Xd",k,0,"R",1]+a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]-b*opt["Xd",k,0,"R",2]+a*opt["Xd",k,0,"R",5]) ],
                       [float(opt["Xd",k,0,"q",0]-b*opt["Xd",k,0,"R",0]-a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]-b*opt["Xd",k,0,"R",1]-a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]-b*opt["Xd",k,0,"R",2]-a*opt["Xd",k,0,"R",5]) ],
                       [float(opt["Xd",k,0,"q",0]+b*opt["Xd",k,0,"R",0]-a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]+b*opt["Xd",k,0,"R",1]-a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]+b*opt["Xd",k,0,"R",2]-a*opt["Xd",k,0,"R",5]) ],
                       ])

    ax.plot([float(opt["Xd",k,0,"q",0]-b*opt["Xd",k,0,"R",0]), float(opt["Xd",k,0,"q",0]-a*opt["Xd",k,0,"R",0])],
        [float(opt["Xd",k,0,"q",1]-b*opt["Xd",k,0,"R",1]), float(opt["Xd",k,0,"q",1]-a*opt["Xd",k,0,"R",1])],
        [float(opt["Xd",k,0,"q",2]-b*opt["Xd",k,0,"R",2]), float(opt["Xd",k,0,"q",2]-a*opt["Xd",k,0,"R",2])],color='black',linewidth = 4)

    tri = a3.art3d.Poly3DCollection([vtx])
    tri.set_color(colors.rgb2hex([.5, .8, .4]))
    tri.set_edgecolor('k')
    tri.set_zorder=1
    ax.add_collection3d(tri)


    # ax.plot([float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0])+10*Faero_E[k][0]/max_normFaero],
    #         [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1])+10*Faero_E[k][1]/max_normFaero],
    #         [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2])+10*Faero_E[k][2]/max_normFaero],color='red',linewidth = 2)
    #

plt.axis('equal')
plt.show()



plt.figure()
plt.plot(tgrid_xa,np.sqrt(val_opt['c']+np.ones(len(val_opt['c']))*params['l']**2))
plt.grid('on')
plt.ylim((299,301))
plt.title('tether length')


plt.figure('power')
plt.plot(tgrid_xa, np.divide(val_opt['power'],1000))
plt.plot(tgrid_xa, opt['Xd',-1,-1,'E']/float(ScalePower)/opt['tf']*np.ones(len(tgrid_xa))/1000)
plt.grid('on')
plt.ylabel('power [kW]')
plt.xlabel('time [s]')
plt.figure('power')
plt.plot()


#
# # FORCE VISUALIZATION
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1,projection='3d')
# k = 0
# F_aero_norm = 10*val_opt['F_aero'][k]/np.linalg.norm(val_opt['F_aero'][k])
# F_drag_norm = 10*val_opt['F_drag'][k]/np.linalg.norm(val_opt['F_drag'][k])
# F_tether_norm = 10*val_opt['F_tether'][k]/np.linalg.norm(val_opt['F_tether'][k])
#
# A = 20
# b = 0.5*0.15*A
# a = 0.5*0.85*A
#
# ax.plot([float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0]+a*opt["Xd",k,0,"R",0])],
#         [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1]+a*opt["Xd",k,0,"R",1])],
#         [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2]+a*opt["Xd",k,0,"R",2])],color='red')
#
#
# ax.plot([float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0]+a*opt["Xd",k,0,"R",3])],
#         [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1]+a*opt["Xd",k,0,"R",4])],
#         [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2]+a*opt["Xd",k,0,"R",5])],color='blue')
#
#
# ax.plot([float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0]+a*opt["Xd",k,0,"R",6])],
#         [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1]+a*opt["Xd",k,0,"R",7])],
#         [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2]+a*opt["Xd",k,0,"R",8])],color='green')
#
# vtx = np.array([
#                    [float(opt["Xd",k,0,"q",0]+b*opt["Xd",k,0,"R",0]+a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]+b*opt["Xd",k,0,"R",1]+a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]+b*opt["Xd",k,0,"R",2]+a*opt["Xd",k,0,"R",5]) ],
#                    [float(opt["Xd",k,0,"q",0]-b*opt["Xd",k,0,"R",0]+a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]-b*opt["Xd",k,0,"R",1]+a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]-b*opt["Xd",k,0,"R",2]+a*opt["Xd",k,0,"R",5]) ],
#                    [float(opt["Xd",k,0,"q",0]-b*opt["Xd",k,0,"R",0]-a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]-b*opt["Xd",k,0,"R",1]-a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]-b*opt["Xd",k,0,"R",2]-a*opt["Xd",k,0,"R",5]) ],
#                    [float(opt["Xd",k,0,"q",0]+b*opt["Xd",k,0,"R",0]-a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]+b*opt["Xd",k,0,"R",1]-a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]+b*opt["Xd",k,0,"R",2]-a*opt["Xd",k,0,"R",5]) ],
#                    ])
#
# # TAIL
# ax.plot([float(opt["Xd",k,0,"q",0]-b*opt["Xd",k,0,"R",0]), float(opt["Xd",k,0,"q",0]-a*opt["Xd",k,0,"R",0])],
#         [float(opt["Xd",k,0,"q",1]-b*opt["Xd",k,0,"R",1]), float(opt["Xd",k,0,"q",1]-a*opt["Xd",k,0,"R",1])],
#         [float(opt["Xd",k,0,"q",2]-b*opt["Xd",k,0,"R",2]), float(opt["Xd",k,0,"q",2]-a*opt["Xd",k,0,"R",2])],color='black',linewidth = 4)
#
# # Wind
# ax.plot(
#             [float(opt["Xd",k,0,"q",0]), 1.1 * float(opt["Xd",k,0,"q",0])],
#             [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1])],
#             [float(opt["Xd",k,0,"q",2])-5.0, float(opt["Xd",k,0,"q",2])-5.]
#             )
#
# # F drag
# ax.plot(
#             [float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0]) + F_drag_norm[0]],
#             [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1]  + F_drag_norm[1])],
#             [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2]) + F_drag_norm[2]],linewidth = 2, color = 'yellow'
#             )
#
# # F tether
# ax.plot(
#             [float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0]) + F_tether_norm[0]],
#             [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1]  + F_tether_norm[1])],
#             [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2]) + F_tether_norm[2]],linewidth = 2, color = 'red'
#             )
#
# # F aero
# ax.plot(
#             [float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0]) + F_aero_norm[0]],
#             [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1]  + F_aero_norm[1])],
#             [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2]) + F_aero_norm[2]],linewidth = 2, color = 'black'
#             )
#
#
# tri = a3.art3d.Poly3DCollection([vtx])
# tri.set_color(colors.rgb2hex([.5, .9, .4]))
# tri.set_edgecolor('k')
# tri.set_zorder=1
# ax.add_collection3d(tri)
#



print "\n\n"
print "Average Power = ", -opt['Xd',-1,-1,'E']/float(ScalePower)/opt['tf'], "  Orbit period = ", opt['tf']
