'''
PLOT the optimal trajectory of states and inputs
of the AWE system in pumping mode
Python Version 2.7 / Casadi version 3.5.5
-
Author: Elena Malz, elena@malz.me
Chalmers, Goeteborg Sweden, 2017, (2020 updated from casadi 2.4.1 to 3.5.5
-
'''

system = 'pump' #or pump

import sys
docpath = r"/Users/elena/Documents/Python/PhDprojects/ElenasDragonfly/"
sys.path.append(r"/Users/elena/Documents/Python/packages/casadi-osx-py27-v3.5.5")
sys.path.append(docpath)

import pickle
from  plotClass import *

# Import parameters and results

if system == 'drag':
    from DRAG.parameters_drag import initial_params as initial_params
    params = initial_params()
    with open(docpath+'DRAG/solution_drag.dat', 'r') as f:
        (val_opt, opt, nk, d) = pickle.load(f)
    with open(docpath+'DRAG/cost_drag.dat', 'r') as f:
        (E_final, Lifting, Tracking, Cost, Reg) = pickle.load(f)
elif system == 'pump':
    from PUMP.values_pump import initial_params as initial_params
    params = initial_params()
    with open(docpath+'PUMP/solution_pump.dat', 'r') as f:
        (val_opt, opt, nk, d) = pickle.load(f)
    with open(docpath+'PUMP/cost_pump.dat', 'r') as f:
        (E_final, Lifting, Tracking, Cost, Reg) = pickle.load(f)


plt.close('all')
plt.ion()
# --------------------------------------------------------

print "\n\n"
print "Average Power = ", -opt['Xd',-1,-1,'E']/float(params['ScalePower'])/opt['tf'], "  Orbit period = ", opt['tf']

p = plots(opt, val_opt, params,nk,d)
tgrid_x  = p.tgrid_x
tgrid_u  = p.tgrid_u
tgrid_xa = p.tgrid_xa
p.plottraj()
p.plotcontrols()
p.plot3Dtraj()
p.costweighting(E_final, Lifting, Tracking, Cost, Reg)
# --------------------------------------------------------
# --- WIND AND KITE SPEED
# --------------------------------------------------------
plt.figure('wind')
plt.subplot(3,1,1)
plt.plot(tgrid_xa,val_opt['windspeed_shear'], label = 'wind speed'); plt.legend()
plt.subplot(3,1,2)
plt.plot(tgrid_xa,val_opt['speed'], label = 'kite speed'); plt.legend()
plt.subplot(3,1,3)
plt.plot(tgrid_xa,val_opt['power'], label = 'power production'); plt.legend()

# --------------------------------------------------------
# --- AoA and sslip
# --------------------------------------------------------
p.draw(tgrid_xa,val_opt['AoA_deg'],tgrid_xa,val_opt['sslip'],'AoA_deg', 'sslip_deg')
# --------------------------------------------------------
# --- CL and CD
# --------------------------------------------------------
p.get_CLCD('wind')
p.savePDF_all()
