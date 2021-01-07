# Airborne Wind Energy Power Optimization 
# ( pumping and drag mode system )

——————Author: Elena Malz, elena@malz.me, Chalmers 2017, updated 2021——————

## Summary Keywords
This is a script that computes the optimal trajectory of an AWE system with the aim of maximizing the power during one orbit.
- Optimal control problem (OCP) of an AWE system including a mathematical model of an dynamic system of the kite. The dynamics are described with DAEs.
- Main control variables are roll, pitch and yaw as well as the tether velocity for pumping mode or the drag force of the onboard turbines for the drag mode type
- The Cost function includes mainly trajectory tracking and power maximisation. Optimal control problem is solved via direct collocation and homotopy strategies.


## Folder Content

### DRAG
* *Main_Dragonly.py*:           Execute program!
* *collocation_drag.py*:        Discretization via collocation
* *aero_drag_.py* :             Aerodynamic model of the rigid body
* *parameters_drag_small.py* :  Parameters for the AWE system
* *parameters_drag.py* :        Parameters for the AWE system
* *`*`.dat* :                  Example solution files of the OCP

### PUMP

* *Main_Pump.py*:               Execute program!
* *collocation_pump.py*:        Discretization via collocation
* *aero_pump.py* :              Aerodynamic model of the rigid body
* *parameters_pump.py* :        Parameters for the AWE system
* *parameters_pump_industry.py*:Parameters for the AWE system
* *`*`.dat* :                   Example solution files of the OCP


### Visualize
* *plotClass.py* :              Class for different plots
* *plot_sol.py*:                Program to plot solutions of drag or pump AWE system


## Needed software
* Python 2.7
* CasADi 3.5.5 (might work with any CasADi > 3.0.0)
* (optional) Other linear solves for IPOPT as e.g. MA27

## Installation
You can get the CasADi package at https://web.casadi.org/.

The solvers one can get from http://www.hsl.rl.ac.uk/ipopt/ and include it as described at https://github.com/casadi/casadi/wiki/Obtaining-HSL.

## Background
This project is part of a dissertation, which can be found at https://research.chalmers.se/en/publication/519020.
If there are any questions please do not hesitate to write an e-mail.
