'''
MAIN Code to run the optimisation of a pumping kite system
based on direct Collocation_pump

Python Version 2.7 / Casadi version 2.4.1
- Author: Elena Malz, Chalmers 2016
'''

from casadi import *
from casadi.tools import *

import numpy as NP
import matplotlib.pyplot as plt
import sys
sys.path.append('dat')
from scipy.linalg import logm, expm
import pickle

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import scipy as sp
from values_pump import initial_params
# from values_pump_Ampyx import initial_params

from aero_Rachel import aero
from Collocation_pump import collocate
import sys
sys.path.append('dat')

# For visualizing
import time
import json
import zmq
start_time = time.time()

plt.close('all')
t = SX.sym('t')

def initial_guess(t, rounds):
    "Creates inital guess for ipopt"
    x_guess = xd()
    inclination    = 30.0*pi/180.0  # the other angle than the one you're thinking of
    dcmInclination = np.array([[np.cos(inclination), 0.0, -np.sin(inclination)],
                               [                0.0, 1.0,                  0.0],
                               [np.sin(inclination), 0.0,  np.cos(inclination)]])
    dtheta = rounds * 2.0 * pi/tf_init
    theta  = t * dtheta
    r      = 0.25 * l       # Circle radius

    angle = SX.sym('angle')
    x_cir = sqrt(l**2 - r**2)
    y_cir = r * cos(angle)
    z_cir = r * sin(angle)
    init_pos_fun = SXFunction('init_pos',[angle],[mul(dcmInclination, veccat([x_cir, y_cir, z_cir]))])
    init_vel_fun = init_pos_fun.jacobian()

    ret = {}
    ret['q']    = init_vel_fun([theta])[1]
    ret['dq']   = init_vel_fun([theta])[0] * dtheta
    ret['w']    = veccat([0.0, 0.0, dtheta])

    norm_vel = np.linalg.norm(ret['dq'])
    norm_pos = np.linalg.norm(ret['q'])

    R0    = ret['dq']/norm_vel
    R2    = ret['q']/norm_pos
    R1    = cross(ret['q']/norm_pos,ret['dq']/norm_vel)
    ret['R'] = vertcat([R0.T, R1.T, R2.T]).T
    ret['dltet'] = 0 #[m/s]
    ret['ltet']  = l
    return ret


def skew(a,b,c):
    " creates skew-symmetric matrix"
    d =  blockcat([[0.,        -c, b],
                   [c,  0.,       -a],
                   [-b,   a,    0.]])
    return d

def get_Output(V,P):
    return outputs(Out_fun([V,P])[0])


# -----------------------------------------------------------
nk = 20     # Control discretization  - for a longer time horizon nk has to be high #(20/40/.. dividable by 4)
d  = 3      # number of polynomial points. Degree of interpolating polynomial yields integration order 5
tf_init = 10.0    # End time initial guess
ScalePower = 1.

# ------------------------
# MODEL PARAMETER
# ------------------------
params = initial_params()
m   = params['mK'] + 1./3*params['mT']    #mass of kite and tether
l   = params['l']               # length of tether
g   = params['g']               # gravity
A   = params['sref']            # area of Kite
J   = params['J']               # Kite Inertia
scale = 1000         # scaling in the dynamics


# --- Declare variables (use scalar graph)
xa  = SX.sym('xa')          # algebraic state
p   = SX.sym('p')           # parameters

# ------------------------
# VARIABLES /  STATES+INPUTS
# ------------------------
xd = struct_symSX([(
                    entry('q', shape = (3,1)),      # position
                    entry('dq', shape = (3,1)),     # velocity of the kite
                    entry('R', shape = (3,3)),      # rotation matrix DCM: transfors from body to fixed frame
                    entry('w', shape = (3,1)),      # angular velocity
                    entry('coeff', shape = (3,1)),  # moment coefficients Cr, Cp, Cy
                    entry('E'),                     # generated energy
                    entry('Drag'),                  # Drag Force
                    entry('ltet'),                  # tether length
                    entry('dltet'),                 # tether reel in/out velocity
                    )])

xddot = struct_symSX([ entry('d'+name, shape = xd[name].shape) for name in xd.keys() ] )
[q, dq, R, w, coeff, E, Drag, ltet,dltet ] = xd[...]


u = struct_symSX([                              # 3 dimensional control input + torque
                  (
                   entry('u', shape = (3,1)),          # artificial force
                   entry('T', shape = (3,1)),          # artificial moments
                   entry('dcoeff', shape = (3,1)),     # actual control input
                   entry('dDrag'),                     # control input (change in drag)
                   entry('ddltet')                    # tether reel in/out acceleration
                   )])

#   generating a structure for the references building the cost function
ref = struct_symSX( [ entry(name, shape = xd[name].shape) for name in xd.keys() ] + \
                    [ entry(name, shape =  u[name].shape) for name in  u.keys() ]
                     )

weights =  struct_symSX(
                        [entry(name, shape = xd[name].shape) for name in xd.keys()]  + \
                        [entry(name, shape =  u[name].shape) for name in  u.keys()]  + \
                        [entry('AoA'),
                        entry('sslip'),
                        ])

p = struct_symSX([(
                   entry('ref', struct = ref),          # stracking reference for cost function
                   entry('weights', struct = weights),  # weights for cost function
                   entry('gam'),                        # homopty variable
                   entry('wind'),
                   entry('z0')                          # roughness factor
                 )])


_,outputs = aero(xd, xa, u, p, params)
outputs['c']  = 0.5*(sum(q**2) - ltet**2 )     # algebraic contstraint
outputs['dc'] = sum(q*dq) - ltet*dltet

out = struct_SX( [entry(name, expr=outputs[name]) for name in outputs] )
out_fun = SXFunction('outputs', [xd , xa, u, p], [out])

# ----------------
# AERODYNAMICS
# ----------------
# m = params['mK'] + 1./3 *   params['tether_density'] * ltet
(Fa, M,  F_tether_scaled, F_drag, F_gravity, Tether_drag), outputs = aero(xd, xa, u,p, params)

wx = skew(w[0],w[1],w[2]) # creating a skrew matrix of the angular velocities

# Rdot = R*w (whereas w is in the skrew symmetric form)
Rconstraint = reshape( xddot['dR'] - mul(xd['R'],wx),9,1 )
TorqueEq    = mul(J,xddot['dw']) + (np.cross(w.T,mul(J,w).T).T - scale*(1-p['gam'])*u['T']) - p['gam']*M # J*wdot +w x J*w = T
DragForce   = -Drag*R[0:3]  # Drag force in kite's x direction

# ------------------------------------------------------------------------------------
#  DYNAMICS of the system in impicit form + Create a structure for the Differential-Algebraic Equation
# ------------------------------------------------------------------------------------
res = vertcat([
               xddot['dq']        - xd['dq'],\
               xddot['dcoeff']    - u['dcoeff'],\
               xddot['dDrag']     - u['dDrag'],\
               m*(xddot['ddq'][0]  + F_tether_scaled[0] - A*(1-p['gam'])*u['u',0]) -  p['gam']*Fa[0] - Tether_drag[0], \
               m*(xddot['ddq'][1]  + F_tether_scaled[1] - A*(1-p['gam'])*u['u',1]) -  p['gam']*Fa[1] - Tether_drag[1], \
               m*(xddot['ddq'][2]  + F_tether_scaled[2] - A*(1-p['gam'])*u['u',2]) -  p['gam']*Fa[2] - Tether_drag[2] + F_gravity   , \
               xddot['dE'] - ScalePower*mul((xa*ltet),dltet), # Note: Don't forget the *m later \
               xddot['dltet']         - xd['dltet'], \
               xddot['ddltet']        - u['ddltet']
              ])

# P = (xa*q)*dl  = (xa *l) * dl = F_tether*speed, since xa is calculated for the tether scaled,
# the final energy has to be multiplied by m as well.

res.append(Rconstraint) # adding R dot to the dynamics
res.append(TorqueEq)    # adding the torque - inertia-eq to the dynamics
res.append(sum(xd['q']*xddot['ddq'])+sum(xd['dq']**2)-sum(xd['ltet']*xddot['ddltet'])-sum(xd['dltet']**2) ) # add tether length as constraint

# System dynamics function (implicit formulation)
dynamics = SXFunction('dynamics', [xd,xddot,xa,u,p],[res])


# --------------------------------
# BUILD LAGRANGE FUNCTION
# --------------------------------

Lagrange_Tracking = 0
Lagrange_Regularisation = 0

# input regularization
for name in set(u.keys()):
    Lagrange_Regularisation += p['weights',name][0]*mul((u[name]-p['ref',name]).T,u[name]-p['ref',name])

Lagrange_Regularisation += p['weights','AoA']*out['AoA']**2
Lagrange_Regularisation += p['weights','sslip']*out['sslip']**2

# Initialization tracking
for name in set(xd.keys())- set(['R','E','Drag']):
    Lagrange_Tracking += p['weights',name][0]*mul((xd[name]-p['ref',name]).T,xd[name]-p['ref',name])
for k in range(9):
    Lagrange_Tracking += reshape(np.linalg.norm(p['weights','R'])*mul((xd['R']-p['ref','R']).T,xd['R']-p['ref','R']),9,1)[k]

Lagrange_Tracking       = SXFunction('lagrange_track', [xd,xa,u,p],[Lagrange_Tracking])
Lagrange_Regularisation = SXFunction('lagrange_reg', [xd,xa,u,p],[Lagrange_Regularisation])

# -----------------------------------------------
# DISCRETIZATION VIA COLLOCATION / SET UP NLP
# -----------------------------------------------
V, P, coll_cstr, continuity_cstr, Output = collocate(xd,xa,u,p,nk,d,dynamics, out_fun,out)
Out_fun = MXFunction('Ofcn',[V,P],[Output])

# --------------------------------------------------
# ADD PATH CONSTRAINTS AND BOUNDARY CONDITIONS
# --------------------------------------------------
DCM_cstr        = [] # meet R.T*R - I = 0
periodic_cstr   = [] # periodic optimisation problem
tether_cstr     = [] # c = 0;
AoA_cstr        = [] # limit angle of attack
sslip_cstr      = [] # limit side slip
speed_cstr      = [] # limit kite speed
E_cstr          = [] # Initial Energy to zero
F_tether_cstr   = [] # maximal allowed tether force

# --- ROTATION MATRIX CONSTRAINT
# rotation matrix DCM has to be orthogonal at every stage. It should be valid R0'R0-I = 0 as well as R0'RN I=0.
# However the constraints, to get in total only 9 constraints, not 18. (1,2)(1,3)(2,3) to the latter an the rest to the former equation.

R0R0 = mul(V['Xd',0,0,'R'].T,V['Xd',0,0,'R'])   - np.eye(3)
R0RN = mul(V['Xd',0,0,'R'].T,V['Xd',-1,-1,'R']) - np.eye(3)

for k in [0,3,4,6,7,8]:
    DCM_cstr.append(R0R0[k])

for k in [1,2,5]:
    DCM_cstr.append(R0RN[k])


# --- PERIODICITY
# add an constraint so that the inital and the final position is the same.
# hereby initial bounds have to be relaxed so that the optimiser decides by itself where to start.

xd_names = xd.keys()
for name in set(xd_names)-set(['R','E','q','dq']):
    periodic_cstr.append( V['Xd',0,0,name]-V['Xd',-1,-1,name] )

periodic_cstr.append( V['Xd',0,0, 'q'] - V['Xd',-1,-1, 'q'] + V['Xd',0,0,'dq']*V['vlift'])
periodic_cstr.append( V['Xd',0,0,'dq'] - V['Xd',-1,-1,'dq'] + V['Xd',0,0, 'q']*V['vlift'])

periodic_cstr = veccat(periodic_cstr)


# -- TETHER LENGTH --- c = 0, at one time point (dc = 0 leads to LICQ problems)
tether_cstr.append(sum(V['Xd',0,0,'q']**2) - V['Xd',0,0,'ltet']**2 )

# --- OUTPUT CONSTRAINTS ----
output_constraints = Output(Out_fun([V,P])[0])

AoA_cstr.append(output_constraints['AoA_deg'])
sslip_cstr.append(output_constraints['sslip_deg'])
E_cstr.append( V['Xd',0,0, 'E'] )
# F_tether_cstr.append(output_constraints['F_tether_norm'])

# STRUCT OF ALL CONSTRAINTS:
g = struct_MX(
              [
               entry('collocation', expr=coll_cstr),
               entry('continuity',  expr=continuity_cstr),
               entry('DCM orthogonality', expr=DCM_cstr),
               entry('periodicity', expr=periodic_cstr),
               entry('tether',      expr = tether_cstr),
               entry('E_cstr',      expr = E_cstr),
               entry('AoA_cstr',    expr = AoA_cstr),
               entry('sslip_cstr',  expr = sslip_cstr),
            #    entry('F_tether_cstr', expr = F_tether_cstr)

            ]
              )

g_fun = MXFunction('g',[V,P],[g])


# --------------------------------
# OBJECTIVE FUNTION
# ------------------------------
Tracking       = 0
Regularisation = 0

for k in range(nk):  # V['XA',k,0] is not same time step as V['Xd',k,0] but same result
    [ftrack] = Lagrange_Tracking([V['Xd',k,0], V['XA',k,0], V['U',k], P['p',k,0]])
    Tracking += ftrack

    [freg] = Lagrange_Regularisation([V['Xd',k,0], V['XA',k,0], V['U',k], P['p',k,0]])
    Regularisation += freg



# free time circle Regularisation
# tracking dissappears slowly in the cost function and Energy maximising appears. at the final step, cost function
# contains maximising energy, lift, SOSC, and regularisation.
E_final             = 1e2 * -V['Xd',-1,-1,'E']   # for maximising final energy

Tracking_Cost       = (1-P['toggle_to_energy']) * Tracking * 1e-3  # Tracking of initial guess
Regularisation_Cost = Regularisation                          # Regularisation of inputs
Lift_Cost           = 0.5*V['vlift']**2 * 1e2                 # Regularisation of inputs
Energy_Cost         = P['toggle_to_energy'] * (E_final/A)/V['tf']
SOSCFix             = 10. * V['Xd',nk/4,0,'q',1]**2

Cost = 0
Cost = (Tracking_Cost + Regularisation_Cost + Lift_Cost + SOSCFix)/float(nk) + Energy_Cost
# Cost += P['tf_LM_regularization']*(V['tf']-P['tf_previous'])**2

# Some functions for ploting the Cost
lift_Cost_fun     = MXFunction('Lift_Cost', [V], [Lift_Cost/float(nk)] )
Energy_Cost_fun   = MXFunction('Energy_Cost', [V,P], [Energy_Cost] )
Tracking_Cost_fun = MXFunction('Tracking_Cost', [V,P], [Tracking_Cost/float(nk)])
Reg_Cost_fun      = MXFunction('Regularisation_Cost', [V,P], [Regularisation_Cost/float(nk)])

totCost_fun       = MXFunction('Cost', [V,P], [Cost])

# --------------
# BOUNDS
# -------------
# lower/upper bounds and initial guess for all variables
vars_lb   = V(-inf)
vars_ub   = V(inf)
vars_init = V()

# Specify inequality constraints
lbg = g()
ubg = g()
lbg['AoA_cstr'] = -11 # in degrees
ubg['AoA_cstr'] =  11
lbg['sslip_cstr'] = -30
ubg['sslip_cstr'] =  30
# ubg['F_tether_cstr'] =  params['F_tether_max']
# lbg['F_tether_cstr'] =  -inf
# # lbg['CL']    =  -inf  # no negative CL, AWE page 336
# ubg['CL']    = inf
# -------------------
# INITIALIZE STATES
# -------------------
tau_roots = collocationPoints(d, 'radau')

for k in range(nk):
    for j in range(d+1):
        t = (k + tau_roots[j])*tf_init/float(nk)
        guess = initial_guess(t,1)

        vars_init['Xd',k,j,'q']  = guess['q']
        vars_init['Xd',k,j,'dq'] = guess['dq']
        vars_init['Xd',k,j,'w']  = guess['w']
        vars_init['Xd',k,j,'R']  = guess['R']
        vars_init['Xd',k,j,'ltet']  = guess['ltet']
        vars_init['Xd',k,j,'dltet'] = guess['dltet']

vars_lb['Xd',:,:,'q',2]  = params['windShearRefAltitude']
# vars_lb['Xd',:,:,'coeff'] = -15*pi/180.
# vars_ub['Xd',:,:,'coeff'] =  15*pi/180.

vars_init['XA',:,:] = 0
vars_lb["XA",:]     = 0   # tension should be positive
vars_ub["XA",:]     = inf

vars_init["U",:] = 0
vars_lb["U",:]   = -inf
vars_ub["U",:]   = inf



# ---------------------
# PARAMETER VALUES
# --------------------

# Build reference parameters, references of cost function should match the initial guess
p_num = P()
# Circle tracking
for name in xd.keys():
    p_num['p',:,:,'ref',name] = vars_init['Xd',:,:,name]

# Weights for cost function
p_num['p',:,:,'weights','q']  = 1.
p_num['p',:,:,'weights','dq'] = 1.
p_num['p',:,:,'weights','R']  = 1.
p_num['p',:,:,'weights','w']  = 1.
p_num['p',:,:,'weights','coeff']  = 1.
p_num['p',:,:,'weights','dltet']  = 10.
p_num['p',:,:,'weights','ltet']  = 10.


p_num['p',:,:,'weights','ddltet']  = 10.
p_num['p',:,:,'weights','u']  = 1e-2
p_num['p',:,:,'weights','T']  = 1e-2
p_num['p',:,:,'weights','dDrag']  = 0.001
p_num['p',:,:,'weights','dcoeff'] = 10.
p_num['p',:,:,'weights','AoA']    = 1.
p_num['p',:,:,'weights','sslip']  = 1.

# p_num['tf_LM_regularization'] = 0.
# p_num['tf_previous']  = tf_init
p_num['p',:,:,'wind'] = params['wind0']
p_num['tf']          = tf_init
p_num['p',:,:,'z0']  = 0.15
vars_init['tf']      = vars_lb['tf'] = vars_ub['tf']  = p_num['tf']


## --------------------
## SOLVE THE NLP
## --------------------

# Allocate an NLP solver
nlp = MXFunction('nlp', nlpIn(x=V, p=P),nlpOut(f=Cost,g=g))

# Set options
opts = {}
opts["expand"] = True
opts["max_iter"] = 1000
opts["tol"] = 1e-8
opts["linear_solver"] = 'ma27'
solver = NlpSolver("solver", "ipopt", nlp, opts)


# ------------------------------------------
#       START LOOP
# ------------------------------------------
# using homotopy for shifting from artificial controls to actual aerodynamic
# forces and actualy control inputs.

Homotopy_step = 0.1
for gamma_value in list(np.arange(0,1.+Homotopy_step,Homotopy_step)):

    p_num['p',:,:,'gam']      = gamma_value
    p_num['toggle_to_energy'] = 0.
    external_extra_text = ['HOMOTOPY SOLVE FOR  GAMMA %.1f' % gamma_value]
    # Initial condition
    arg = {}
    arg['x0']   = vars_init

    # Bounds on x
    arg['lbx'] = vars_lb
    arg['ubx'] = vars_ub

    # Bounds on g
    arg['lbg'] = lbg
    arg['ubg'] = ubg

    arg['p']   = p_num   # hand over the parameters to the solver

    # Solve the problem
    print '   '
    print 'Solve for gamma:   ',p_num['p',0,0,'gam'] #PARAMETER value for homotopy
    print '   '
    res = solver(arg)
    stats = solver.getStats()
    assert stats['return_status'] in ['Solve_Succeeded']
    print '   '
    print 'Solved for gamma:  ',p_num['p',0,0,'gam'] #PARAMETER value for homotopy
    print '   '

    arg['lam_x0'] = res['lam_x']

    # Retrieve the solution
    opt = V(res['x'])

    # update initial guess with NLP solution
    vars_init = opt


# Save final optimum for initial guess
init_opt = opt
vars_init1 = opt

# -----------------------------------------------------------------------------
# POWER OPTIMISATION
# -----------------------------------------------------------------------------
# Using homopty for changing cost function.
# Shifting from tracking to power optimisation
print "#####################################################"
print "#####################################################"
print "#####################################################"
print "#########                                   #########"
print "#########    STARTING POWER OPTIMIZATION    #########"
print "#########                                   #########"
print "#####################################################"
print "#####################################################"
print "#####################################################"

arg['x0'] = vars_init

#Ensure no fictitious forces
p_num['p',:,:,'gam']      = 1.

# Change weight on tether vel and acc
p_num['p',:,:,'weights','dltet']  = 0.
p_num['p',:,:,'weights','ltet']   = 0.
p_num['p',:,:,'weights','ddltet'] = 10.
p_num['p',:,:,'weights','sslip']  = 0.
p_num['p',:,:,'weights','AoA']    = 0.

Homotopy_step = 0.1
toggle_table = list(np.arange(Homotopy_step,1.+ Homotopy_step,Homotopy_step))
for toggle_value in toggle_table:

    #Update toggle value
    p_num['toggle_to_energy'] = toggle_value
    arg['p']  = p_num
    external_extra_text = ['POWER OPTIMIZATION; TOGGLE %.1f - FIXED TIME' % toggle_value]
    # Solve the problem
    print "Solve for toggle =", toggle_value
    res = solver(arg)

    # Retrieve the solution, re-assign as new guess
    arg['x0']            = res['x']
    arg['lam_x0']        = res['lam_x']
    # p_num['tf_previous'] = V(res['x'])['tf']
    arg['p']             = p_num

    #Report some stuff...
    print "Solved for toggle =", toggle_value, " Period = ", float(V(res['x'])['tf'])



opt = V(res['x'])

outputs = Output(Out_fun([V,P])[0])
val_init = get_Output(vars_init,p_num)
val_opt = get_Output(opt,p_num)

# with open('solution_pump_f.dat','w') as f:
#     pickle.dump((val_opt, opt, nk, d),f)

# with open('init_f.dat','w') as f:
#     pickle.dump((val_init, vars_init1, nk, d),f)
#

# switch = raw_input("free final time? (j/n)\n\n")
# if switch == 'j':

print "#####################################################"
print "#####################################################"
print "#####################################################"
print "#########                                   #########"
print "#########          OPEN FINAL TIME          #########"
print "#########                                   #########"
print "#####################################################"
print "#####################################################"
print "#####################################################"

# update initial guess with NLP solution
vars_init = opt

# initial guess on the spiral
rounds = 3
for k in range(nk):
    for j in range(d+1):
        t = (k + tau_roots[j])*opt['tf']/float(nk)
        guess = initial_guess(t,rounds)

        vars_init['Xd',k,j,'q']  = guess['q']
        vars_init['Xd',k,j,'dq'] = guess['dq']
        vars_init['Xd',k,j,'w']  = guess['w']
        vars_init['Xd',k,j,'R']  = guess['R']
        vars_init['Xd',k,j,'ltet']  = guess['ltet']
        vars_init['Xd',k,j,'dltet'] = guess['dltet']



arg['x0']   = vars_init

# FREE FINAL TIME
vars_lb['tf'] = -inf
vars_ub['tf'] = inf

arg['lbx'] = vars_lb
arg['ubx'] = vars_ub

# Levenberg-Marquardt regularization on the orbit period
# p_num['tf_previous'] = V(res['x'])['tf']
# p_num['tf_LM_regularization'] = 0.
# p_num['p',:,:,'weights','ddltet']  = 50.
arg['p']  = p_num
external_extra_text = ['RELEASE TIME - final solve']

# Solve
res = solver(arg)

#------------------------------
# RECEIVE SOLUTION  & SAVE DATA
#------------------------------
opt = V(res['x'])


outputs = Output(Out_fun([V,P])[0])
val_init = get_Output(vars_init,p_num)
val_opt = get_Output(opt,p_num)

with open('solution_pump.dat','w') as f:
    pickle.dump((val_opt, opt, nk, d),f)

with open('init.dat','w') as f:
    pickle.dump((val_init, vars_init1, nk, d),f)


[E_final]   = Energy_Cost_fun([opt,p_num])
[Lifting]   = lift_Cost_fun([opt])
[Tracking]  = Tracking_Cost_fun([opt,p_num])
[Cost]      = totCost_fun([opt,p_num])
[Reg]       = Reg_Cost_fun([opt,p_num])

with open('cost.dat', 'w') as f:
    pickle.dump((E_final, Lifting, Tracking, Cost, Reg), f)


# --------------------------------------
# PRINT OUT ....
# --------------------------------------
print "\n\n\n"
print "Average Power = ", opt['Xd',-1,-1,'E']*m/float(ScalePower)/opt['tf'], "  Orbit period = ", opt['tf']

end_time = time.time()
time_taken = end_time - start_time
print time_taken

# --------------------------------------
# Check cost function....
# --------------------------------------

plt.ion()
plt.figure('Cost function weighting')
ax = plt.subplot(111)
track0 = ax.bar(1,np.array(Tracking),0.1, color = 'r')
regu0 = ax.bar(1+0.1,np.array(Reg),0.1, color = 'b')
lifting0 = ax.bar(1+0.2,np.array(Lifting),0.1, color = 'k')
energy0 = ax.bar(1+0.3,np.array(E_final*-1),0.1, color = 'g')
cost0 = ax.bar(1+0.4,np.array(Cost),0.1, color = 'y')
ax.legend((track0,regu0,lifting0,energy0,cost0), (('tracking','regularisation','lifting','energy*-1','total cost')))
plt.grid('on')
# ax.set_ylim([0,3000])
ax.set_xlim([1,1.5])

plt.show()


#
# # -----------------------
# ## CHECK SOSC
# # -----------------------
#
# # computes the nullspace (equivalent to the casadi-nullspace command)
#
# def null(A, V_shape, eps=1e-4, ):
#     nullshape = V_shape[0]-A.shape[0]
#     [u, s, vh] = sp.linalg.svd(A)       # python gives v.T back so a=u*s*vh
#
#     # Check if Ax = 0, or rather dgopt[0]*Null=0. Then v[..] is the nullspace
#     zeros = np.dot(A,vh[A.shape[0]+1:,:].T)
#     maxzero = max(abs(np.concatenate(zeros)))
#     nullspace = vh[A.shape[0]:,:].T
#
#     return nullspace, s, u, vh, zeros
#
# # Make function out of equality constraints
# geq = struct_MX(
#               [
#                entry('collocation', expr=coll_cstr),
#                entry('continuity',  expr=continuity_cstr),
#                entry('DCM orthogonality', expr=DCM_cstr),
#                entry('periodicity', expr=periodic_cstr),
#                entry('tether',      expr = tether_cstr),
#                entry('E_cstr', expr = E_cstr)
#             ]
#               )
#
# geq_fun = MXFunction('geq',[V,P],[geq])
# geq_jac = geq_fun.jacobian()
#
# # compute hessian at the solution
# hess_fun                      = solver.hessLag()
# [Hopt, fopt, gopt, dLx, dLy]  = hess_fun([res['x'], p_num, 1, res['lam_g']])
# # Hopt: Hessian at the optimum, fopt: cost at the optimum,
# # gopt: constraints at the optimum, dLx: gradient of lagrangian with respect to the states (sensitivity),
# # dLy: Langrangian sensitivity wrt the multipliers
#
# dgopt = geq_jac([res['x'], p_num])                         # evaluate equality constraint-deriv at solution
# [N_dgopt, sv, _, _, zeros]     = null(dgopt[0],V.shape)                       # compute nullspace
#
# # LICQ
# print 'LICQ_check; smallest singular value', sv[-1], 'biggest sv',  sv[0]
#
# # SOSC
# redH        = mul([N_dgopt.T, Hopt, N_dgopt])              # compute reduced hessian
# [eigs,eigv] = np.linalg.eig(redH)
#
# print 'Eigenvalues: ' , eigs  # Eigenvalues should be >0 in order to fullfill SOSC
# print 'Lift variable', opt['vlift']
#
# print np.max(val_opt['power'])- np.min(val_opt['power'])
