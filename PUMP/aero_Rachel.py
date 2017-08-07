'''
simple AERODYNAMIC MODEL of AWE system (pumping mode)
takes states and inputs and creates aerodynamic forces and moments
dependent on the position of the kite.
Aerodynamic coefficients are assumptions.
Python Version 2.7 / Casadi version 2.4.1
- Author: Elena Malz, Chalmers 2016
'''
import casadi as ca
import casadi.tools as ca
import numpy as np
from numpy import pi

def aero(xd, xa, u, p, params):
    [q, dq, R, w, coeff, E, Drag, ltet, dltet ] = xd[...]

    # troposphere density changed
    gasconst = 287.05                 # gas constant
    temp = (15+273)-6.5/1000*q[-1]    # T(h)
    pressure = (1010-0.103*q[-1])*100 # hPa
    rho = pressure / (gasconst*temp)  # air density
    params['rho'] = rho

    # WIND SHEAR AND APPARENT VELOCITY
    windShearRefAltitude = 5
    # windspeed_shear = params['wind0']*np.log(q[-1]/windShearRefAltitude)
    # windspeed_shear = p['wind']*(q[-1]/windShearRefAltitude)**0.2
    windspeed_shear = p['wind']*(q[-1]/windShearRefAltitude)**p['z0']
    v_app = dq - np.array([windspeed_shear,0,0])
    # v_app = dq - np.array([11,0,0])  # constant wind

    # calculate angle of attack and side slip (convert apparent velocity to body frame)
    AoA       = -ca.mul(R[6:9].T,(v_app))/ca.mul(R[0:3].T,(v_app))
    sslip     = ca.mul(R[3:6].T,(v_app))/ca.mul(R[0:3].T,(v_app))

    # get moment coefficients dependent on AoA,sslip and speed
    CF, CM = aero_coeffs(AoA, sslip, v_app, w,  coeff , params)

    F_aero = 0.5 * params['rho'] * params['sref'] * np.linalg.norm(v_app) * (ca.cross(CF[2]*v_app,R[3:6]) + CF[0]*(v_app) )
    reference_lengths = ca.vertcat([params['bref'], params['cref'], params['bref']])
    M_aero = 0.5 * params['rho'] * params['sref'] *  np.linalg.norm(v_app)**2 * CM  * reference_lengths

    F_drag   = -Drag * R[0:3]   # Drag in opposite x-direction of kite (Props)
    F_tether = q*xa             # Force of the tether at the kite
    m = params['mK'] + 1./3*params['mT']
    # m   = params['mK'] + 1./3*params['tether_density']*ltet    #mass of kite and tether

    F_gravity = m*params['g']

    # Tether Drag
    # defined in + x direction , therefore minus sign
    C_tet        = 0.4
    Tether_drag  =  - 1./8 * params['rho'] * params['tether_diameter'] * C_tet * ltet* np.linalg.norm(v_app)**2 * R[0:3]
    # Tether_drag  =  - 1./6 *  rho * params['tether_diameter'] * C_tet * ltet* np.linalg.norm(v_app)**2 * R[0:3]

    # mechanical data of winch
    r_winch    = 0.1
    omega_mech = dltet/r_winch
    T_mech     = xa*ltet*r_winch
    outputs = {}
    outputs['v_app']    = v_app
    outputs['speed']    = np.linalg.norm(v_app)
    outputs['windspeed_shear'] = windspeed_shear
    outputs['AoA']      = AoA
    outputs['sslip']    = sslip
    outputs['AoA_deg']  = AoA  * 180/pi
    outputs['sslip_deg']= sslip * 180/pi
    outputs['CL']       = CF[2]
    outputs['CD']       = -CF[0]
    outputs['F_aero']   = F_aero
    outputs['Lift_aero'] =  0.5 * params['rho'] * params['sref'] * np.linalg.norm(v_app) * (ca.cross(CF[2]*v_app,R[3:6]))
    outputs['Drag_aero'] =  0.5 * params['rho'] * params['sref'] * np.linalg.norm(v_app) * (CF[0]*(v_app) )
    outputs['F_drag']   = F_drag
    outputs['F_tether_scaled'] = F_tether   # This is scaled so Force/kg
    outputs['F_tether'] = F_tether*m
    outputs['F_tether_norm'] = np.linalg.norm(F_tether*m)
    # outputs['mtether']  = m
    # outputs['F_gravity'] = F_gravity
    outputs['M_aero']   = M_aero
    outputs['power']    = ca.mul((xa*ltet*m),dltet)
    outputs['Tether_drag'] = Tether_drag
    outputs['omega_mech']  = omega_mech
    outputs['T_mech']      = T_mech
    outputs['pressure']    = pressure
    outputs['temp']        = temp
    outputs['rho']         = rho
    outputs['P_el'] = 0.95*outputs['power']
    return (F_aero, M_aero, F_tether, F_drag, F_gravity, Tether_drag), outputs

def aero_coeffs(alpha, beta, v_app, omega,  phi , params):
    ''' gives all aerodynamic coefficients '''

    alphaDeg = alpha * 180./np.pi
    CL = params['CL0'] + params['CLalpha'] * alphaDeg
    CD = params['CD0'] + params['CDalpha'] * alphaDeg + params['CDalphaSq'] * alphaDeg**2.0 + 2.*beta**2 + 0.2*alpha**2
    # CL = (params['CL0']-0.05) + (params['CLalpha']+0.02) * alphaDeg
    # CD = params['CD0'] + (params['CDalpha']+0.001) * alphaDeg + (params['CDalphaSq']-0.0002) * alphaDeg**2.0 + 2.*beta**2

    # CD = 0.067 - 0.009845* alphaDeg + .0006284 * alphaDeg **2  + 2.*beta**2
    # CL = 0.32 + 0.06808 * alphaDeg


    CFx_0 = - CD   # theoretical adding rotor drag but no rotors
    CFy_0 = 0.     #2*pi*0.1 * beta            #num_pylons * CY_b * pylon_sref/params['sref']
    CFz_0 = CL

    CMx_0 = 0.
    CMy_0 = 0.
    CMz_0 = 0.

    # pqr - DAMPING

    CFx_pqr = 0.
    CFy_pqr = 0.
    CFz_pqr = 0.

    omega_hat = omega/(2.*np.linalg.norm(v_app))
    omega_hat[0] *= params['bref']
    omega_hat[1] *= params['cref']
    omega_hat[2] *= params['bref']

    p = omega_hat[0]
    q = omega_hat[1]
    r = omega_hat[2]

    # roll moment, about ehat1
    Cl = params['Clp'] * p + params['Clr'] * r + params['Clbeta'] * beta
    # pitch moment, about ehat2
    Cm = params['Cmq'] * q + params['Cmalpha'] * alpha
    # yaw moment, about ehat3
    Cn = params['Cnp'] * p + params['Cnr'] * r  + params['Cnbeta'] * beta

    CMx_pqr = Cl
    CMy_pqr = Cm
    CMz_pqr = Cn

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

    CF_wind = ca.vertcat([CFx, CFy, CFz]) # in fixed frame
    CM_cad = ca.vertcat([CMx, CMy, CMz])  # in body frame

    return CF_wind, CM_cad
