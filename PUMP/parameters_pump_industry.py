'''
Collected parameters of the pumping Kite system
Python Version 3.8 / CasADi version 3.5.5

-
@Author: Elena Malz, elena@malz.me
Chalmers, Goeteborg Sweden, 2017
(2020: updated from Python 2.7/casADi 2.4.1 to Python 3.8/casADi 3.5.5)
-
'''
from collections import OrderedDict
import numpy as np
from numpy import pi

def initial_params():
    params = OrderedDict()

    params['ScalePower'] = 1.
    # kite mass [kg]
    params['mK'] = 36.8   # 50 * 25 [AWE book p.15]
    # acceleration due to gravity [m/s^]
    params['g'] = 9.81
    # density of air [kg/m^3]
    params['rho'] = 1.2  # with 1.255 is actually different !?
    # airframe moment of inertia [kg*m^2]
    params['J'] = np.array([[25,   0.0,   -4.7],
                            [  0.0, 32,   0.0],
                            [  -0.47,   0.0, 56.0]])

    # tether natural length [m]
    params['l'] = 300.
    # tether mass [kg]
    params['mT'] = 0.0046 * params['l']
    # aerodynamic reference area [m^2]
    params['sref'] = 3.
    # aerodynamic reference span [m]
    params['bref'] = 5.5
    # aerodynamic reference chord [m]
    params['cref'] = params['sref']/params['bref']
    # reference wind at ground [m/s]
    params['wind0'] = 6.0
    # tether diameter [m]
    params['tether_diameter'] = 0.0025
    # tether density [kg/m]
    params['tether_density'] = 0.0046
    # altitude where wind shear starts
    params['windShearRefAltitude'] = 5.

    params['alphaMaxDeg']   = 10.0 # deg. separation indicated somewhere between 12 and 15 degrees.
    params['CL0']           = 0.3455
    params['CLalpha']       = 0.04808 #* 180. / pi # per rad
    params['CD0']           = 0.02875
    params['CDalpha']       = -0.003561 #* 180. / pi # per rad
    params['CDalphaSq']     = 0.0006284 #* (180. / pi)*2 # per rad**2
    params['Clp']           = -0.48 * 180. / pi # per rad
    params['Clr']           = 0.01 * 180. / pi  # per rad
    params['Clbeta']        = -0.0008 * 180. / pi # per rad
    params['Cmalpha']       = -0.005 * 180. / pi # per rad
    params['Cmq']           = -9. # per rad
    params['Cnp']           = -0.11 * 180. / pi # per rad
    params['Cnr']           = -0.03 # per rad
    params['Cnbeta']        = 0.0003 * 180. / pi # per rad

    return params
