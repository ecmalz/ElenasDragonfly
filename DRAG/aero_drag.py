
'''
simple AERODYNAMIC MODEL of AWE system (drag mode)
Aerodynamic coefficients are assumptions.
Python Version 3.8 / CasADi version 3.5.5

-
@Author: Elena Malz, elena@malz.me
Chalmers, Goeteborg Sweden, 2017
(2020: updated from Python 2.7/casADi 2.4.1 to Python 3.8/casADi 3.5.5)
-
'''
import casadi as ca
import casadi.tools as ca
import sys

import numpy as np
from numpy import pi

def aero(xd, xa, p, params):
    [q, dq, R, w, coeff, E, Drag ] = xd[...]

    # wind vector & apparent velocity
    windShearRefAltitude = 5
    # windspeed_shear = params['wind0']*np.log(q[-1]/windShearRefAltitude)
    windspeed_shear = p['wind']*(q[-1]/windShearRefAltitude)**p['z0']
    v_app           = dq - np.array([windspeed_shear,0,0])
    # v_app = dq - np.array([11,0,0])

    # calculate angle of attack and side slip (convert apparent velocity to body frame)
    AoA       = -ca.mtimes(R[6:9].T,(v_app))/ca.mtimes(R[0:3].T,(v_app))
    sslip     = ca.mtimes(R[3:6].T,(v_app))/ca.mtimes(R[0:3].T,(v_app))

    CF, CM = aero_coeffs(AoA, sslip, v_app, w,  coeff , params)

    F_aero = 0.5 * params['rho'] * params['sref'] * ca.norm_2(v_app) * (ca.cross(CF[2]*v_app,R[3:6]) + CF[0]*(v_app) )
    reference_lengths = ca.vertcat(params['bref'], params['cref'], params['bref'])
    M_aero = 0.5 * params['rho'] * params['sref'] *  ca.norm_2(v_app)**2 * CM  * reference_lengths

    F_drag   = -Drag * R[0:3]   # Drag in opposite x-direction of kite (Props)
    F_tether = q*xa             # Force of the tether at the kite
    m = params['mK'] + 1./3*params['mT']
    F_gravity = m*params['g']

    # Tether Drag
    # defined in + x direction , therefore minus sign
    C_tet        = 0.4
    Tether_drag  =  - 1./8 * params['rho'] * params['tether_diameter'] * C_tet * params['l'] * ca.norm_2(v_app)**2 * R[0:3]
    # Tether_drag  =  - 1./6 * params['rho'] * params['tether_diameter'] * C_tet * params['l'] * ca.norm_2(v_app)**2 * R[0:3]

    outputs = {}
    outputs['v_app']    = v_app
    outputs['speed']    = ca.norm_2(v_app)
    outputs['windspeed_shear'] = windspeed_shear
    outputs['AoA']      = AoA
    outputs['sslip']    = sslip
    outputs['AoA_deg']  = AoA  * 180/pi
    outputs['sslip_deg']= sslip * 180/pi
    outputs['CL']       = CF[2]
    outputs['CD']       = -CF[0]
    outputs['F_aero']   = F_aero
    outputs['F_drag']   = F_drag
    outputs['F_tether_scaled'] = F_tether   # This is scaled so Force/kg
    outputs['F_tether'] = F_tether*m
    outputs['M_aero']   = M_aero
    outputs['power']    = ca.mtimes(F_drag.T,dq)
    outputs['Tether_drag'] = Tether_drag
    return (F_aero, M_aero, F_tether, F_drag, F_gravity, Tether_drag), outputs

def aero_coeffs(alpha, beta, v_app, omega,  phi , params):
    ''' gives all aerodynamic coefficients '''


    CL = 0.8   * alpha/ (10*pi/180)    # = 20*pi/180, but if i put in correct number, solutions are changed ..
    CD = 0.008 + 0.04*(alpha/(10*pi/180))**2 + 2.*beta**2

    CFx_0 = - CD   # theoretical adding rotor drag but no rotors
    CFy_0 = 0      #2*pi*0.1 * beta          #num_pylons * CY_b * pylon_sref/params['sref']
    CFz_0 = CL

    CMx_0 = 0.
    CMy_0 = 0. #0.1 * alpha   #+ 0.75 * alpha_tail
    CMz_0 = 0.      #0.75 * beta_tail

    # pqr - DAMPING
    omega_hat = omega/(2.*ca.norm_2(v_app))
    omega_hat[0] *= params['bref']
    omega_hat[1] *= params['cref']
    omega_hat[2] *= params['bref']

    CFx_pqr = 0.
    CFy_pqr = 0.
    CFz_pqr = 0.

    CMx_pqr = 0.
    CMy_pqr = 0.
    CMz_pqr = 0.
    CMx_pqr = ca.mtimes(omega_hat.T,np.array([-10.,   0.0,  - 1e-2]))
    CMy_pqr = ca.mtimes(omega_hat.T,np.array([ 0.0,  0.0,   0.0]))
    CMz_pqr = ca.mtimes(omega_hat.T,np.array([ - 1e-1,  0.0,   0.0]))

    # surfaces
    CFx_surfs = 0.
    CFy_surfs = 0.
    CFz_surfs = 0.
    CMx_surfs = 0.1 * phi[0]
    CMy_surfs = 0.1 * phi[1]
    CMz_surfs = 0.1 * phi[2]


    CFx = CFx_0 + CFx_pqr + CFx_surfs
    CFy = CFy_0 + CFy_pqr + CFy_surfs
    CFz = CFz_0 + CFz_pqr + CFz_surfs
    CMx = CMx_0 + CMx_pqr + CMx_surfs
    CMy = CMy_0 + CMy_pqr + CMy_surfs
    CMz = CMz_0 + CMz_pqr + CMz_surfs

    CF_wind = ca.vertcat(CFx, CFy, CFz) # in fixed frame
    CM_cad = ca.vertcat(CMx, CMy, CMz)  # in body frame

    return CF_wind, CM_cad
