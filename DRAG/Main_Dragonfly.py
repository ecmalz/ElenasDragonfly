'''
MAIN Code to run the optimisation of a drag mode kite system
based on direct Collocation

Python Version 3.8 / CasADi version 3.5.5

-
@Author: Elena Malz, elena@malz.me
Chalmers, Goeteborg Sweden, 2017
(2020: updated from Python 2.7/casADi 2.4.1 to Python 3.8/casADi 3.5.5)
-
'''


import sys
import os
casadi_path = r"/Users/elena/Documents/Python/packages/casadi-osx-py38-v3.5.5"
if not os.path.exists(casadi_path):
    print('Casadi package path is wrong!')
    sys.exit()

sys.path.append(casadi_path)
from casadi import *
from casadi.tools import *

import numpy as NP
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm

#import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import scipy as sp
from parameters_drag import initial_params
from aero_drag import aero
# from aero_Dragonfly_Rachel import aero

from collocation_drag import collocate

# For visualizing
import time
#import json
#import zmq
start_time = time.time()


plt.close('all')

def initial_guess(t):
    x_guess = xd()
    inclination    = 30.0*pi/180.0  # the other angle than the one you're thinking of
    dcmInclination = np.array([[np.cos(inclination), 0.0, -np.sin(inclination)],
                               [                0.0, 1.0,                  0.0],
                               [np.sin(inclination), 0.0,  np.cos(inclination)]])
    dtheta = 2.0 * pi/tf_init
    theta  = t * dtheta
    r      = 0.25 * l

    angle = SX.sym('angle')
    x_cir = sqrt(l**2 - r**2)
    y_cir = r * cos(angle)
    z_cir = r * sin(angle)
    init_pos_fun = Function('init_pos',[angle],[mtimes(dcmInclination, veccat(x_cir, y_cir, z_cir))])
    init_vel_fun = init_pos_fun.jacobian()

    ret = {}
    ret['q']    = init_pos_fun(theta)
    ret['dq']   = init_vel_fun(theta,0) * dtheta
    ret['w']    = veccat(0.0, 0.0, dtheta)

    norm_vel = norm_2(ret['dq'])
    norm_pos = norm_2(ret['q'])

    R0    = ret['dq']/norm_vel
    R2    = ret['q']/norm_pos
    R1    = cross(ret['q']/norm_pos,ret['dq']/norm_vel)
    ret['R'] = vertcat(R0.T, R1.T, R2.T).T
    return ret

def skew(a,b,c):
    d =  blockcat([[0.,        -c, b],
                   [c,  0.,       -a],
                   [-b,   a,    0.]])

    return d

def get_Output(V,P):
    return outputs(Out_fun(V,P))



# -----------------------------------------------------------

nk = 20    # Control discretization     % for a longer time horizon nk has to be high
d  = 3       # Degree of interpolating polynomial yields integration order 5
tf_init = 9.0    # End time initial guess
ScalePower = 1e-3

# ------------------------
# MODEL
# ------------------------
params = initial_params()
m = params['mK'] + 1./3*params['mT']
l   = params['l']      # length of tether
g   = params['g']
rho = params['rho']         # air density
A   = params['sref']   # area of Kite

scale = 1000 # scaling in the dynamics
J = params['J']  # Inertia


# --- Declare variables (use scalar graph)
xa  = SX.sym('xa')          # algebraic state
p   = SX.sym('p')       # parameters


# ---VARIABLES /  STATES+INPUTS ------

xd = struct_symSX([(
                    entry('q', shape = (3,1)),      # position
                    entry('dq', shape = (3,1)),     # velocity of the kite
                    entry('R', shape = (3,3)),      # rotation matrix DCM: transfors from body to fixed frame
                    entry('w', shape = (3,1)),      # angular velocity
                    entry('coeff', shape = (3,1)),  # moment coefficients Cr, Cp, Cy
                    entry('E'),                     # energy
                    entry('Drag')
                    )])

xddot = struct_symSX([ entry('d'+name, shape = xd[name].shape) for name in xd.keys() ] )
[q, dq, R, w, coeff, E, Drag ] = xd[...]


u = struct_symSX([                              # 3 dimensional control input + torque
                  (
                   entry('u', shape = (3,1)),
                   entry('T', shape = (3,1)),
                   entry('dcoeff', shape = (3,1)),
                   entry('dDrag')
                   )
                  ])

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
                   entry('ref', struct = ref),
                   entry('weights', struct = weights),
                   entry('wind'),
                   entry('gam'),
                   entry('z0')
                 )])

_,outputs = aero(xd, xa, p, params)
outputs['c']  = ( sum1(q**2) - l**2 )
outputs['dc'] = sum1(q*dq)


out = struct_SX( [entry(name, expr=outputs[name]) for name in outputs] )
out_fun = Function('outputs', [xd , xa, p], [out]) # Has been SXFunction  in casadi 2

# -------------
# AERODYNAMICS
# -------------


(Fa, M,  F_tether_scaled, F_drag, F_gravity, Tether_drag), outputs = aero(xd, xa, p, params)
windShearRefAltitude = 5.

wx = skew(w[0],w[1],w[2]) # creating a skrew matrix of the angular velocities


# Rdot = R*w (whereas w is in the skrew symmetric form)
Rconstraint = reshape( xddot['dR'] - mtimes(xd['R'],wx),9,1 ) # J*wdot +w x J*w = T
TorqueEq = mtimes(J,xddot['dw']) + (cross(w.T,mtimes(J,w).T).T - scale*(1-p['gam'])*u['T']) - p['gam']*M
DragForce = -Drag*R[0:3]

# ------------------------------------------------------------------------------------
#  DYNAMICS of the system - Create a structure for the Differential-Algebraic Equation
# ------------------------------------------------------------------------------------
res = vertcat(
               xddot['dq']        - xd['dq'],\
               xddot['dcoeff']    - u['dcoeff'],\
               xddot['dDrag']     - u['dDrag'],\
               m*(xddot['ddq'][0]  + F_tether_scaled[0] - A*(1-p['gam'])*u['u',0]) -  p['gam']*Fa[0] - F_drag[0] - Tether_drag[0], \
               m*(xddot['ddq'][1]  + F_tether_scaled[1] - A*(1-p['gam'])*u['u',1]) -  p['gam']*Fa[1] - F_drag[1] - Tether_drag[1], \
               m*(xddot['ddq'][2]  + F_tether_scaled[2] - A*(1-p['gam'])*u['u',2]) -  p['gam']*Fa[2] - F_drag[2] - Tether_drag[2]+ F_gravity   , \
               xddot['dE'] - ScalePower*mtimes(F_drag.T,dq)
              )


# adding R dot to the dynamics
res = veccat(res,Rconstraint) # reshape matrix to a list

# adding the torque - inertia-eq to the dynamics
res = veccat(res,TorqueEq)

# add tether length as constraint
res = veccat(res, sum1(xd['q']*xddot['ddq'])+sum1(xd['dq']**2))

# System dynamics function (implicit formulation)
dynamics = Function('dynamics', [xd,xddot,xa,u,p],[res])


# --------------------------------
# BUILD LAGRANGE FUNCTION
# --------------------------------

Lagrange_Tracking = 0
Lagrange_Regularisation = 0

# input regularization
for name in set(u.keys()):
    Lagrange_Regularisation += p['weights',name][0]*mtimes((u[name]-p['ref',name]).T,u[name]-p['ref',name])

Lagrange_Regularisation += p['weights','AoA']*out['AoA']**2
# Lagrange_Regularisation += (1 + mtimes(xd['R'][:,2].T,xd['q'] / sqrt(mtimes(xd['q'].T,xd['q']))))

# Initialization tracking
for name in set(xd.keys())- set(['R','E','Drag']):
    Lagrange_Tracking += p['weights',name][0]*mtimes((xd[name]-p['ref',name]).T,xd[name]-p['ref',name])
for k in range(9):
    Lagrange_Tracking += reshape(p['weights','R'][0]*mtimes((xd['R']-p['ref','R']).T,xd['R']-p['ref','R']),9,1)[k]


Lagrange_Tracking       = Function('lagrange_track', [xd,xa,u,p],[Lagrange_Tracking])
Lagrange_Regularisation = Function(  'lagrange_reg', [xd,xa,u,p],[Lagrange_Regularisation])

# -----------------------------
# DISCRETIZATION / SET UP NLP
# -----------------------------

V, P, coll_cstr, continuity_cstr, Output = collocate(xd,xa,u,p,nk,d,dynamics, out_fun,out)
Out_fun = Function('Ofcn',[V,P],[Output])

# ADD PATH CONSTRAINTS AND BOUNDARY CONDITIONS


DCM_cstr        = [] # meet R.T*R - I = 0
periodic_cstr   = [] # periodic optimisation problem
tether_cstr     = [] # c = 0;
E_cstr          = []  # Initial Energy to zero

AoA_cstr        = [] # limit angle of attack
sslip_cstr      = [] # limit side slip

# --- ROTATION MATRIX CONSTRAINT
# rotation matrix DCM has to be orthogonal at every stage. It should be valid R0'R0-I = 0 as well as R0'RN I=0.
# However the constraints, to get in total only 9 constraints, not 18. (1,2)(1,3)(2,3) to the latter an the rest to the former equation.

R0R0 = mtimes(V['Xd',0,0,'R'].T,V['Xd',0,0,'R'])   - np.eye(3)
R0RN = mtimes(V['Xd',0,0,'R'].T,V['Xd',-1,-1,'R']) - np.eye(3)

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

periodic_cstr = veccat(*periodic_cstr)


# -- TETHER LENGTH --- c = 0, dc = 0 at one time point
tether_cstr.append(sum1(V['Xd',0,0,'q']**2) - l**2 )

# --- OUTPUT CONSTRAINTS ----
output_constraints = Output(Out_fun(V,P))

AoA_cstr.append(output_constraints['AoA_deg'])
sslip_cstr.append(output_constraints['sslip_deg'])
E_cstr.append( V['Xd',0,0, 'E'] )

# Struct of all constraints:
g = struct_MX(
              [
               entry('collocation', expr=coll_cstr),
               entry('continuity',  expr=continuity_cstr),
               entry('DCM orthogonality', expr=DCM_cstr),
               entry('periodicity', expr=periodic_cstr),
               entry('tether',      expr = tether_cstr),
               entry('E_cstr', expr = E_cstr)
            ]
              )

h = struct_MX([
               entry('AoA_cstr',      expr = AoA_cstr),
               entry('sslip_cstr',    expr = sslip_cstr),
            ])


cstr_struct = struct_MX([entry('g', expr = g, struct = g),
                         entry('h', expr = h,  struct = h)])

gh = vertcat(g,h)
gh = cstr_struct(gh)
g_fun = Function('g',[V,P],[g])
h_fun = Function('h',[V,P],[h])

# --------------------------------
# Objective function
# ------------------------------
Tracking       = 0
Regularisation = 0

for k in range(nk):  # V['XA',k,0] is not same time step as V['Xd',k,0] but same result
    ftrack = Lagrange_Tracking(V['Xd',k,0], V['XA',k,0], V['U',k], P['p',k,0])
    Tracking += ftrack

    freg = Lagrange_Regularisation(V['Xd',k,0], V['XA',k,0], V['U',k], P['p',k,0])
    Regularisation += freg



# free time circle Regularisation
# tracking dissappears slowly in the cost function and Energy maximising appears. at the final step, cost function
# contains maximising energy, lift, SOSC, and regularisation.
E_final             = 10. * V['Xd',-1,-1,'E']    # for maximising final energy

Tracking_Cost       = (1-P['toggle_to_energy']) * Tracking #* 1e-3  # Tracking of initial guess
Regularisation_Cost = Regularisation                          # Regularisation of inputs
Lift_Cost           = 0.5*V['vlift']**2 #* 1e2                      # Regularisation of inputs
Energy_Cost         = P['toggle_to_energy'] * (E_final/A)/V['tf']
# SOSCFix             = 10. * V['Xd',5,0,'q',1]**2              # Fix one point to avoid SOSC (y to 0, highest point)
SOSCFix             = 10. * V['Xd',int(nk/4),0,'q',1]**2

Cost = 0
Cost = (Tracking_Cost + Regularisation_Cost + Lift_Cost + SOSCFix)/float(nk) + Energy_Cost

# Cost += 10. * V['Xd',5,0,'q',1]**2              # Fix one point to avoid SOSC (y to 0, highest point)
# # Cost += P['tf_LM_regularization']*(V['tf']-P['tf_previous'])**2

# ------ Plot Cost ------

lift_Cost_fun     = Function('Lift_Cost', [V], [Lift_Cost/float(nk)] )
Energy_Cost_fun   = Function('Energy_Cost', [V,P], [Energy_Cost] )
Tracking_Cost_fun = Function('Tracking_Cost', [V,P], [Tracking_Cost/float(nk)])
Reg_Cost_fun      = Function('Regularisation_Cost', [V,P], [Regularisation_Cost/float(nk)])

totCost_fun       = Function('Cost', [V,P], [Cost])



# --------------
# BOUNDS
# -------------

vars_lb   = V(-inf)
vars_ub   = V(inf)
vars_init = V()

# Concatenate constraints
lbg = cstr_struct()
ubg = cstr_struct()
lbg['h','AoA_cstr'] = -15  # in degrees
ubg['h','AoA_cstr'] =  15
lbg['h','sslip_cstr'] = -30
ubg['h','sslip_cstr'] =  30


# --------------
# INITIALIZE STATES
# -------------
tau_roots = collocation_points(d, 'radau')
tau_roots = veccat(0, tau_roots)

for k in range(nk):
    for j in range(d+1):
        t = (k + tau_roots[j])*tf_init/float(nk)
        guess = initial_guess(t)

        vars_init['Xd',k,j,'q']  = guess['q']
        vars_init['Xd',k,j,'dq'] = guess['dq']
        vars_init['Xd',k,j,'w']  = guess['w']
        vars_init['Xd',k,j,'R']  = guess['R']

# print  mtimes(vars_init["Xd",0,0,'R'].T,vars_init["Xd",0,0,'R'])
# print 'det:  ', np.linalg.det(vars_init["Xd",0,0,'R'])

vars_lb['Xd',:,:,'q',2] = params['windShearRefAltitude']

vars_lb['Xd',:,:,'coeff'] = -15*pi/180.
vars_ub['Xd',:,:,'coeff'] =  15*pi/180.
## INTIALIZE CONTROLS /  ALGEBRAIC VARIABLE / FINAL TIME
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
    # Circle
for name in xd.keys():
    p_num['p',:,:,'ref',name] = vars_init['Xd',:,:,name]

# wind velocities
p_num['p',:,:,'wind'] = params['wind0']
p_num['p',:,:,'weights','q']  = 1.
p_num['p',:,:,'weights','dq'] = 1.
p_num['p',:,:,'weights','R']  = 1.
p_num['p',:,:,'weights','w']  = 1.
p_num['p',:,:,'weights','coeff']  = 0.01

p_num['p',:,:,'weights','u']  = 0.01
p_num['p',:,:,'weights','T']  = 0.01
p_num['p',:,:,'weights','dDrag']  = 0.001
p_num['p',:,:,'weights','dcoeff'] = 10.
p_num['p',:,:,'weights','AoA']    = 1.
# p_num['p',:,:,'weights','sslip']  = 0.

# p_num['tf_previous'] = tf_init
p_num['p',:,:,'wind'] = params['wind0']
p_num['p',:,:,'z0']   = 0.15
p_num['tf'] = tf_init
vars_init['tf'] =  vars_lb['tf'] = vars_ub['tf']  = p_num['tf']


## --------------------
## SOLVE THE NLP
## --------------------

# Allocate an NLP solver
#nlp = MXFunction('nlp', nlpIn(x=V, p=P),nlpOut(f=Cost,g=gh))
nlp = {'x':V, 'p':P, 'f':Cost, 'g':gh}



# Set options
opts = {}
opts["expand"] = True
opts["ipopt.max_iter"] = 1000
opts["ipopt.tol"] = 1e-8
# opts["ipopt.linear_solver"] = 'ma27'
# opts["ipopt.linear_solver"] = 'ma57'

#solver = NlpSolver("solver", "ipopt", nlp, opts)
solver = nlpsol("solver", "ipopt", nlp, opts)


# ------------------------------------------
#       START LOOP
# ------------------------------------------
# using homotopy for shifting from artificial controls to actual aerodynamic
# forces and actualy control inputs.

Homotopy_step = 0.1
for gamma_value in list(np.arange(0,1.+Homotopy_step,Homotopy_step)):

    p_num['p',:,:,'gam'] = gamma_value
    print (gamma_value)
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
    print ('   ')
    print ( 'Solve for gamma:   ',p_num['p',0,0,'gam']) #PARAMETER value for homotopy
    print ('   ')
    res = solver(**arg)
    stats = solver.stats()
    assert stats['return_status'] in ['Solve_Succeeded']
    print ('   ')
    print ('Solved for gamma:  ',p_num['p',0,0,'gam']) #PARAMETER value for homotopy
    print ('   ')

    arg['lam_x0'] = res['lam_x']

    # -----------------------------------------------------------------------------
    # Retrieve SOLUTION
    # -----------------------------------------------------------------------------

    # Retrieve the solution
    opt = V(res['x'])

    # -----------------------------------------------------------------------------
    # UPDATE INITIAL GUESS AND PARAMETERS
    # -----------------------------------------------------------------------------

    # update initial guess with NLP solution

    vars_init = opt


# SAVE FINAL OPTIMUM FOR INITIAL GUESS
init_opt = opt


# -----------------------------------------------------------------------------
# POWER OPTIMISATION
# -----------------------------------------------------------------------------
# Using homopty for changing cost function.
# Shifting from tracking to power optimisation
print ("#####################################################")
print ("#####################################################")
print ("#####################################################")
print ("#########                                   #########")
print ("#########    STARTING POWER OPTIMIZATION    #########")
print ("#########                                   #########")
print ("#####################################################")
print ("#####################################################")
print ("#####################################################")

#Attribute previous guess
arg['x0'] = vars_init

#Ensure no fictitious forces
p_num['p',:,:,'gam']      = 1.

#tf_iterations = [] #record orbit periods

Homotopy_step = 0.1
toggle_table = list(np.arange(Homotopy_step,1.+ Homotopy_step,Homotopy_step))

for toggle_value in toggle_table:

    #Update toggle value
    p_num['toggle_to_energy'] = toggle_value
    arg['p']  = p_num
    external_extra_text = ['POWER OPTIMIZATION; TOGGLE %.1f - FIXED TIME' % toggle_value]
    # Solve the problem
    print( "Solve for toggle =", toggle_value)
    res = solver(**arg)

    # Retrieve the solution, re-assign as new guess
    arg['x0']            = res['x']
    arg['lam_x0']        = res['lam_x']
    # p_num['tf_previous'] = V(res['x'])['tf']
    arg['p']             = p_num

    #Report some stuff...
    print ("Solved for toggle =", toggle_value, " Period = ", float(V(res['x'])['tf']))


opt = V(res['x'])

print ("#####################################################")
print ("#####################################################")
print ("#####################################################")
print ("#########                                   #########")
print ("#########           OPEN FINAL TIME         #########")
print ("#########                                   #########")
print ("#####################################################")
print ("#####################################################")
print ("#####################################################")

vars_lb['tf'] = 0.
vars_ub['tf'] = inf

arg['lbx'] = vars_lb
arg['ubx'] = vars_ub

#vLevenberg-Marquardt regularization on the orbit period
# p_num['tf_previous'] = V(res['x'])['tf']

external_extra_text = ['RELEASE TIME - final solve']
res = solver(**arg)

print ("Period = ", V(res['x'])['tf'])

#------------------------------
# RECEIVE SOLUTION  & SAVE DATA
#------------------------------
opt = V(res['x'])


outputs = Output(Out_fun(V,P))

val_init = get_Output(vars_init, p_num)
val_opt = get_Output(opt,p_num)

import pickle
with open('solution_drag.dat','wb') as f:
    pickle.dump((val_opt, opt, nk, d),f)

with open('init_drag.dat','wb') as f:
    pickle.dump((val_init, vars_init, nk, d),f)

# dyn = g(g_fun([opt,p_num])[0])['collocation']
# with open('dynamics_opt.dat', 'w') as f:
#     pickle.dump(dyn,f)

E_final   = Energy_Cost_fun(opt,p_num)
Lifting      = lift_Cost_fun(opt)
Tracking  = Tracking_Cost_fun(opt,p_num)
Cost      = totCost_fun(opt,p_num)
Reg       = Reg_Cost_fun(opt,p_num)

with open('cost_drag.dat', 'wb') as f:
    pickle.dump((E_final, Lifting, Tracking, Cost, Reg), f)




# --------------------------------------
# PRINT OUT ....
# --------------------------------------
print ("\n\n\n")
print ("Average Power = ", -opt['Xd',-1,-1,'E']/float(ScalePower)/opt['tf'], "  Orbit period = ", opt['tf'])

end_time = time.time()
time_taken = end_time - start_time
print('Done! The computation took ' + str(time_taken)+ 's.')
print('If you want to plot the solution, please check out and run  >> Visualize/plot_sol.py << ')


# -----------------------
## CHECK SOSC
# -----------------------

# computes the nullspace (equivalent to the casadi-nullspace command)

def null(A, V_shape, eps=1e-4, ):
    nullshape = V_shape[0]-A.shape[0]
    [u, s, vh] = sp.linalg.svd(A)       # python gives v.T back so a=u*s*vh

    # Check if Ax = 0, or rather dgopt[0]*Null=0. Then v[..] is the nullspace
    zeros = np.dot(A,vh[A.shape[0]+1:,:].T)
    maxzero = max(abs(np.concatenate(zeros)))
    nullspace = vh[A.shape[0]:,:].T

    return nullspace, s, u, vh, zeros

def computeHessian():
    # Make function out of equality constraints
    geq = struct_MX(
                  [
                   entry('collocation', expr=coll_cstr),
                   entry('continuity',  expr=continuity_cstr),
                   entry('DCM orthogonality', expr=DCM_cstr),
                   entry('periodicity', expr=periodic_cstr),
                   entry('tether',      expr = tether_cstr),
                   entry('E_cstr', expr = E_cstr)
                ]
                  )

    geq_fun = Function('geq',[V,P],[geq])
    geq_jac = geq_fun.jacobian()

    # compute hessian at the solution
    hess_fun                      = solver.hessLag()
    [Hopt, fopt, gopt, dLx, dLy]  = hess_fun([res['x'], p_num, 1, res['lam_g']])
    # Hopt: Hessian at the optimum, fopt: cost at the optimum,
    # gopt: constraints at the optimum, dLx: gradient of lagrangian with respect to the states (sensitivity),
    # dLy: Langrangian sensitivity wrt the mul

    dgopt = geq_jac([res['x'], p_num])                         # evaluate equality constraint-deriv at solution
    [N_dgopt, sv, _, _, zeros]     = null(dgopt[0],V.shape)                       # compute nullspace

    # LICQ
    print ('LICQ_check; smallest singular value', sv[-1], 'biggest sv',  sv[0])

    # SOSC
    redH        = mtimes([N_dgopt.T, Hopt, N_dgopt])              # compute reduced hessian
    [eigs,eigv] = np.linalg.eig(redH)

    # print 'Eigenvalues: ' , eigs  # Eigenvalues should be >0 in order to fullfill SOSC
    # print 'Lift variable', opt['vlift']
    # print np.max(val_opt['power'])- np.min(val_opt['power'])
