'''
PLOT the optimal trajectory of states and inputs
of the AWE system in pumping mode
Python Version 2.7 / Casadi version 2.4.1
- Author: Elena Malz, Chalmers 2016
'''

from casadi import *
from casadi.tools import *
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
from values_pump import initial_params
# from values_pump_Ampyx import initial_params

import sys
# sys.path.append('../../Data/pump_dat')
# sys.path.append('../')   # Dragonfly
import fcts

# ###########
# import sys
# sys.path.append('..')
# from CreateWind import windpolyy, windpolyx
# ############

params = initial_params()



m = params['mK'] + 1./3*params['mT']


plt.close('all')
ScalePower = 1.#e-3

with open('solution_pump.dat', 'r') as f:
    (val_opt, opt, nk, d) = pickle.load(f)

with open('init.dat', 'r') as f:
    (val_init, vars_init, nk, d) = pickle.load(f)

with open('cost.dat', 'r') as f:
    (E_final, Lifting, Tracking, Cost, Reg)= pickle.load(f)


tau_root = collocationPoints(d,'radau')

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


# -------------------------------------------------------
# AoA, SS AND CL/ CD (for real values check book p. 263)
# --------------------------------------------------------
plt.figure('angles and CL/CD')
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
# plt.plot(tgrid_xa, [np.linalg.norm(L)/(np.linalg.norm(D)*np.linalg.norm(TD)) for L,D,TD in zip(val_opt['Lift_aero'],val_opt['Drag_aero'],val_opt['Tether_drag'])])
plt.plot(tgrid_xa, [np.linalg.norm(L)/(np.linalg.norm(D)) for L,D in zip(val_opt['CL'],val_opt['CD'])])
# plt.plot(tgrid_xa, [np.linalg.norm(L)/(np.linalg.norm(D)) for L,D in zip(val_opt['Lift_aero'],val_opt['Drag_aero'])])

plt.grid('on')
plt.title('L/D')
# --------------------------------------------------------
# POSITION , SPEED, ANGULAR VEL
# --------------------------------------------------------
plt.figure()
plt.clf()
for k in range(3):
    plt.subplot(3,3,k+1)
    plt.plot(tgrid_x,opt["Xd",horzcat,:,horzcat,:,'q',k].T)
    plt.plot(tgrid_x,vars_init["Xd",horzcat,:,horzcat,:,'q',k].T)
    plt.title('q'+str(k))
    plt.grid('on')
    plt.subplot(3,3,k+4)
    plt.plot(tgrid_x,opt["Xd",horzcat,:,horzcat,:,'dq',k].T)
    plt.plot(tgrid_x,vars_init["Xd",horzcat,:,horzcat,:,'dq',k].T)
    plt.title('dq'+str(k))
    plt.grid('on')
    plt.subplot(3,3,k+7)
    plt.plot(tgrid_x,opt["Xd",horzcat,:,horzcat,:,'w',k].T)
    plt.plot(tgrid_x,vars_init["Xd",horzcat,:,horzcat,:,'w',k].T)
    plt.title('w'+str(k))
    plt.grid('on')
# --------------------------------------------------------
# FICTIOUS AND REAL CONTROLS
# --------------------------------------------------------
plt.figure()
for k in range(3):
    plt.subplot(4,3,k+1)
    plt.step(tgrid_u,[round(np.array(opt['U',veccat,:,'u', k])[o]) for o in range(0,nk)])
    plt.step(tgrid_u,[round(np.array(vars_init['U',veccat,:,'u', k])[o]) for o in range(0,nk)])
    plt.title('homotopy_accels_v'+str(k))
    plt.grid('on')

    plt.subplot(4,3,k+4)
    plt.step(tgrid_u,[round(np.array(opt['U',veccat,:,'T', k])[o]) for o in range(0,nk)])
    plt.step(tgrid_u,vars_init['U',horzcat,:,'T', k].T)
    plt.title('homotopy_accels_omega'+str(k))
    plt.grid('on')

    plt.subplot(4,3,k+7)
    # plt.plot(tgrid_x,opt['Xd',horzcat,:,horzcat,:,'coeff',k].T*np.ones(tgrid_x.shape[0])*180/pi)
    plt.plot(tgrid_x,opt['Xd',horzcat,:,horzcat,:,'coeff',k].T)
    plt.plot(tgrid_x,vars_init['Xd',horzcat,:,horzcat,:,'coeff',k].T)
    plt.title('coeff'+str(k))
    plt.grid('on')

    plt.subplot(4,3,k+10)
    plt.step(tgrid_u,opt['U',veccat,:,'dcoeff',k])
    plt.step(tgrid_u,vars_init['U',veccat,:,'dcoeff',k])
    plt.title('dcoeff'+str(k))
    plt.grid('on')

# --------------------------------------------------------
# FORCES
# --------------------------------------------------------
Ft_max = pi/2 *  (params['tether_diameter']*1000/2.)**2 * 1073.4
plt.figure()
plt.subplot(3,2,1)
plt.plot(tgrid_xa, [np.linalg.norm(val_opt['F_aero'][k]) for k in range(0,nk*d) ])
plt.grid('on'), plt.title('F_aero')
plt.subplot(3,2,2)
plt.plot(tgrid_xa, [np.linalg.norm(val_opt['Lift_aero'][k]) for k in range(0,nk*d) ])
plt.grid('on'), plt.title('lift force [N]')
plt.subplot(3,2,3)
plt.plot(tgrid_xa, [np.linalg.norm(val_opt['Drag_aero'][k]) for k in range(0,nk*d) ])
plt.grid('on'), plt.title('drag force [N]')
ax = plt.subplot(3,2,4)
ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.plot(tgrid_xa, [np.linalg.norm(val_opt['Tether_drag'][k]) for k in range(0,nk*d) ])
plt.grid('on'), plt.title(['Tether_drag '])
plt.subplot(3,2,5)
plt.plot(tgrid_xa, [val_opt['F_aero'][k][2] for k in range(0,nk*d) ])
plt.grid('on'), plt.title(['F_aero z direction, and gravity'])
plt.subplot(3,2,6)
plt.plot(tgrid_xa, [val_opt['F_tether_norm'][k] for k in range(0,nk*d) ])
#plt.plot(tgrid_xa, params['F_tether_max'] * np.ones(nk*d),'r')
plt.grid('on'), plt.title(['F_tether'])
# --------------------------------------------------------
# POWER
# --------------------------------------------------------
plt.figure('Power')
plt.plot(tgrid_xa, [val_opt['power'][k]/1000. for k in range(0,nk*d)] )
plt.title('OUTPUT - POWER [kW]')

plt.grid('on')

# --------------------------------------------------------
#  3D PLOT /  TRAJECTORIES
# --------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(2,2,1,projection='3d')
ax.plot(np.array(opt["Xd",horzcat,:,horzcat,:,'q',0])[0],np.array(opt["Xd",horzcat,:,horzcat,:,'q',1])[0],np.array(opt["Xd",horzcat,:,horzcat,:,'q',2])[0],'-')
ax.plot(np.array(vars_init["Xd",horzcat,:,horzcat,:,'q',0])[0],np.array(vars_init["Xd",horzcat,:,horzcat,:,'q',1])[0],np.array(vars_init["Xd",horzcat,:,horzcat,:,'q',2])[0],'-')
ax.scatter(0,0,0,'r')
ax.plot([0, opt["Xd",0,0,'q',0]],[0, opt["Xd",0,0,'q',1]],[0,opt["Xd",0,0,'q',2]])
ax.scatter([opt["Xd",3,0,'q',0]],[opt["Xd",3,0,'q',1]],[opt["Xd",3,0,'q',2]],'g')
#ax.set_zlim3d(0.8, 1.1)
plt.axis('equal')

plt.subplot(2,2,2)
plt.plot(np.array(opt["Xd",horzcat,:,horzcat,:,'q',0])[0],np.array(opt["Xd",horzcat,:,horzcat,:,'q',1])[0],'-')
plt.plot(np.array(vars_init["Xd",horzcat,:,horzcat,:,'q',0])[0],np.array(vars_init["Xd",horzcat,:,horzcat,:,'q',1])[0],'-')
plt.title('x-y')
plt.grid('on')

plt.subplot(2,2,3)
plt.plot(np.array(opt["Xd",horzcat,:,horzcat,:,'q',1])[0],np.array(opt["Xd",horzcat,:,horzcat,:,'q',2])[0])
plt.plot(np.array(vars_init["Xd",horzcat,:,horzcat,:,'q',1])[0],np.array(vars_init["Xd",horzcat,:,horzcat,:,'q',2])[0],'-')
plt.title('y-z')
plt.grid('on')

plt.subplot(2,2,4)
plt.plot(np.array(opt["Xd",horzcat,:,horzcat,:,'q',0])[0],np.array(opt["Xd",horzcat,:,horzcat,:,'q',2])[0],'-')
plt.plot(np.array(vars_init["Xd",horzcat,:,horzcat,:,'q',0])[0],np.array(vars_init["Xd",horzcat,:,horzcat,:,'q',2])[0],'-')
plt.title('x-z')

plt.axis('equal')
plt.grid('on')
plt.show()

# --------------------------------------------------------
# LoD FIGURE DEPENDENT ON ANGLE OF ATTACK
# --------------------------------------------------------
beta = 0
AR = params['bref'] / params['cref']
AR = 10
plt.figure()
L   = []
D   = []
LoD = []
sslip = 0.0
alpha_list = np.arange(0,22,0.01)
for alphaDeg in alpha_list:
    # CD = 0.067 - 0.009845* alphaDeg + .0006284 * alphaDeg **2  + 2.*beta**2
    # CL = 0.32 + 0.06808 * alphaDeg
    CL = params['CL0'] + params['CLalpha'] * alphaDeg
    CD = params['CD0'] + params['CDalpha'] * alphaDeg + params['CDalphaSq'] * alphaDeg**2.0 + 0.2*alphaDeg*pi/180**2
    L.append(CL)
    D.append(CD)

    LoD.append(CL/CD)

ax = plt.subplot(1,3,1)
ax.get_yaxis().get_major_formatter().set_useOffset(False)

plt.subplot(1,3,1); plt.plot(alpha_list,L)
plt.grid('on')
plt.subplot(1,3,2); plt.plot(alpha_list,D)
plt.grid('on')
plt.title('L,D,LoD')
plt.subplot(1,3,3); plt.plot(alpha_list,LoD)
plt.grid('on')

# --------------------------------------------------------
# WINDSHEAR, SPEED and ELEVATION ANGLE
# --------------------------------------------------------

elev_angle = []
for k in range(nk):
    for j in range(d+1):

        elev_angle.append(180/pi* np.arccos(np.sqrt(opt['Xd',k,j,'q',0]**2 + opt['Xd',k,j,'q',1]**2) /
                        np.sqrt(sum(opt['Xd',k,j,'q']**2))))
h2 = np.arange(5,1000,10)

plt.figure()
plt.subplot(2,2,1)
plt.plot(tgrid_xa,val_opt['windspeed_shear'])
plt.plot(tgrid_x,params['wind0']*(np.array(opt["Xd",horzcat,:,horzcat,:,'q',2].T)/params['windShearRefAltitude'])**0.2)
plt.grid('on')
plt.title('wind speed')
plt.subplot(2,2,2)
plt.plot(tgrid_xa,val_opt['speed'])
plt.title('kite speed')
plt.grid('on')
plt.subplot(2,2,3)
plt.plot(tgrid_x,elev_angle)
plt.title('elevation angle')
plt.grid('on')
# plt.subplot(2,2,4)
# plt.plot(tgrid_xa,val_opt['windy'])
# plt.title('wind in y direction')
# plt.plot(windpolyy([h2[k] for k in range(0,len(h2))]) , h2)
# plt.plot(params['wind0']*(h2/params['windShearRefAltitude'])**0.2, h2, '-')
# plt.title('wind shear')
plt.grid('on')

# --------------------------------------------------------
# KITE FLYING
# --------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')

# normFaero = []
# for k in np.arange(0,nk,nk/10):
#     normFaero.append(np.linalg.norm(Faero_E[k]))
# max_normFaero = max(normFaero)

for k in np.arange(0,nk,1):

#    b = 0.1*A
#    a = 0.3*A
    A = 13
    # b = 0.5*0.2*A
    # a = 0.5*A
    a = params['bref']
    b = params['cref']

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

    ax.plot(np.array(opt["Xd",horzcat,:,horzcat,:,'q',0])[0],
            np.array(opt["Xd",horzcat,:,horzcat,:,'q',1])[0],
            np.array(opt["Xd",horzcat,:,horzcat,:,'q',2])[0],linestyle = '--', color = 'grey',)


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


# plt.axis('equal')
plt.show()
# --------------------------------------------------------
#  CONSTRAINT, TETHER LENGTH and TETHER ACCELERATION
# --------------------------------------------------------
plt.figure()
plt.subplot(2,2,1)
plt.plot(tgrid_xa,val_opt['c'])
plt.plot(tgrid_xa,val_opt['dc'])
plt.legend(['c','cdot'])
plt.grid('on')
plt.title('tether length constraint')
plt.subplot(2,2,2)
plt.plot(tgrid_x,opt['Xd',horzcat,:,horzcat,:,'ltet'].T)
plt.plot(tgrid_x,vars_init['Xd',horzcat,:,horzcat,:,'ltet'].T)
plt.grid('on')
plt.title('tether length')
plt.subplot(2,2,3)
plt.plot(tgrid_x,opt['Xd',vertcat,:,vertcat,:,'dltet'])
plt.plot(tgrid_x,vars_init['Xd',veccat,:,veccat,:,'dltet'])
plt.grid('on')
plt.title('tether speed')
plt.subplot(2,2,4)
plt.plot(tgrid_u,opt['U',vertcat,:,'ddltet'])
plt.plot(tgrid_u,vars_init['U',veccat,:,'ddltet'])
plt.grid('on')
plt.title('tether acc')

# # --------------------------------------------------------
# # WINCH SPEED AND TORQUE
# # --------------------------------------------------------
# m_winch = 150 #kg
# r_winch = 0.9
# I = 0.5*m_winch*r_winch**2
#
# omega_mech = np.divide(opt['Xd',veccat,:,veccat,1:,'dltet'],r_winch)
# acc_mech = np.divide(opt['U',:,'ddltet'],r_winch)
# T_mech = opt['Xd',veccat,:,veccat,1:,'ltet']*r_winch*opt['XA',veccat,:,veccat,:]
# T_inertia = I*acc_mech
#
#
# plt.figure('WINCH SPEED AND TORQUE')
# plt.subplot(3,1,1)
# # plt.plot(tgrid_xa, val_opt['omega_mech'])
# plt.plot(tgrid_xa,omega_mech*(60./2/pi/r_winch))
# plt.title('omega_mech')
# plt.grid('on')
# plt.subplot(3,1,2)
# plt.plot(tgrid_xa, val_opt['T_mech'])
# plt.plot(tgrid_u,T_inertia)
# plt.title('Torque_mech')
# plt.grid('on')
# plt.subplot(3,1,3)
# plt.plot(tgrid_xa, val_opt['P_el'])
# plt.title('P_el')
# plt.grid('on')
#
# plt.figure('torque/omega')
# plt.plot(omega_mech,T_mech, 'o')
# plt.grid('on')

# --------------------------------------------------------
#  PRESSURE DENSITY AND TEMPERATURE
# --------------------------------------------------------

plt.figure('pressure densitiy and temp')
plt.subplot(3,1,1)
plt.plot(tgrid_xa, val_opt['pressure'])
plt.title('pressure')
plt.grid('on')
plt.subplot(3,1,2)
plt.plot(tgrid_xa, val_opt['temp'])
plt.title('temp')
plt.grid('on')
plt.subplot(3,1,3)
plt.plot(tgrid_xa, val_opt['rho'])
plt.title('rho')
plt.grid('on')



# From Book values. surf-Kite = 30, twing kite = 97
Fout = []
for k in range(nk*d):
    Fout.append(val_opt['CL'][k]**3 / val_opt['CD'][k]**2)

FoutMAX = max(Fout)
print 'F_out_max:   ', FoutMAX, '\n Book: values between 30 and 97'

# --------------------------------------------------------
# WINDSHEAR
# --------------------------------------------------------
hrange = range(0,2000,10)
minh = min(opt['Xd',vertcat,:,vertcat,:,'q',2])
maxh = max(opt['Xd',vertcat,:,vertcat,:,'q',2])
operationrange = range(minh,maxh,10)
windshear = [params['wind0']*(k/params['windShearRefAltitude'])**0.15 for k in hrange]
operation = [params['wind0']*(k/params['windShearRefAltitude'])**0.15 for k in operationrange]
plt.figure()
plt.plot(windshear,hrange, '--')
plt.plot(operation,operationrange, 'r', lw = 2)
plt.grid('on')

# # --------------------------------------------------------
# # Publication plots
# # --------------------------------------------------------
# axisfont = {'size':'28'}
#
# fig = plt.figure()
# plt.subplot(1,2,1)
# plt.plot(np.array(opt["Xd",horzcat,:,horzcat,:,'q',0])[0],np.array(opt["Xd",horzcat,:,horzcat,:,'q',1])[0],'-')
# plt.xlabel('x', **axisfont)
# plt.ylabel('y', **axisfont)
# plt.locator_params(nbins=4)
# plt.tick_params(labelsize=25)
# plt.grid('on')
#
# plt.subplot(1,2,2)
# plt.plot(np.array(opt["Xd",horzcat,:,horzcat,:,'q',0])[0],np.array(opt["Xd",horzcat,:,horzcat,:,'q',2])[0],'-')
# plt.ylabel('z', **axisfont)
# plt.xlabel('x', **axisfont)
# plt.locator_params(nbins=4)
# plt.grid('on')
# plt.tick_params(labelsize=25)
#


# #--- PLOT COST ---
# Regularisation = 0; Tracking       = 0; Reg = []; Track = []
# for k in range(nk):
#     [freg] = Lagrange_Regularisation([opt['Xd',k,0], opt['XA',k,0], opt['U',k], p_num['p',k,0]])
#     Regularisation += freg
#     Reg.append(freg)
#     [ftrack] = Lagrange_Tracking([opt['Xd',k,0], opt['XA',k,0], opt['U',k], p_num['p',k,0]])
#     Tracking += ftrack
#     Track.append(ftrack)
#
# [E_final]= Energy_fun([opt])
# [lift]= lift_fun([opt])
#
# plt.figure()
#
# plt.subplot(222)
# plt.plot(range(nk), Reg)
# plt.title('Regularisation')
# plt.grid('on')
# plt.subplot(224)
# plt.plot(range(nk), Track)
# plt.title('Tracking')
# plt.grid('on')
#
# ax = plt.subplot(121)
# track0 = ax.bar(1,np.array(Tracking*1e-5),0.1, color = 'r')
# regu0 = ax.bar(1+0.1,np.array(Regularisation),0.1, color = 'b')
# lift0 = ax.bar(1+0.2,np.array(lift),0.1, color = 'k')
# energy0 = ax.bar(1+0.3,np.array(E_final*-1),0.1, color = 'g')
# ax.legend((track0,regu0,lift0,energy0), (('tracking*1e5','regularisation','lift','energy*-1')))
# plt.grid('on')
# #ax.set_ylim([0,3000])
# plt.show()




# ## --- ROTATION MATRIX ---
# plt.figure()
# plt.clf()
# for i in range(3):
#     for j in range(3):
#         plt.subplot(3,3,j+3*i+1)
# #        print j+3*i+1
#         plt.plot(tgrid_x,opt["Xd",horzcat,:,horzcat,:,"R",i+3*j].T)
#         plt.plot(tgrid_x,vars_init["Xd",horzcat,:,horzcat,:,"R",i+3*j].T)
#
#         plt.title('R'+str(i+1)+str(j+1))
#         plt.grid('on')




#

# # ------- FORCE VISUALIZATION -------
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


print "Average Power = ", opt['Xd',-1,-1,'E']*m/float(ScalePower)/opt['tf'], "  Orbit period = ", opt['tf']
plt.show()
