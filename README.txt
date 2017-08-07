————
AWE OPTIMIZATION CODE ( PUMPING / DRAG MODE )
Python Version 2.7 / Casadi version 2.4.1
Author: Elena Malz, elenama@chalmers.se, Chalmers 2017
——————

Optimal control problem of an AWE system; mathematical model of the kite is described as DAE; Main control variables are roll, pitch and yaw as well as the rope velocity for pumping mode or the drag force of the onboard turbines for drag mode.
The Cost function includes mainly trajectory tracking and power maximisation. Optimal control problem is solved via direct collocation and homotopy strategies. 

Folder contains:

Main_pump.py   - Execute!
aero_Rachel.py - Aerodynamic model 
Collocation.py — Collocation code
values_pump.py - includes model parameters
fcts.py        - contains different functions used in the code
plot_solution_pump.py — plots different figures of states and controls

