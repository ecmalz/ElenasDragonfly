'''
COLLOCATION Code (direct collocation)
takes variables and the number of nodes/collocation points
creates NLP variables
creates necessary collocation and continutiy constraints
Python Version 2.7 / Casadi version 2.4.1
- Author: Elena Malz, Chalmers 2016
'''
import casadi as ca
import casadi.tools as ca
import sys

import numpy as NP
from numpy import pi

def collocate(xd,xa,u,p,nk,d,dynamics, out_fun,out):
    # -----------------------------------------------------------------------------
    # Collocation setup
    # -----------------------------------------------------------------------------

    # Choose collocation points
    tau_root = ca.collocationPoints(d,'radau')

    # Size of the finite elements
    h = 1./nk

    # Coefficients of the collocation equation
    C = NP.zeros((d+1,d+1))

    # Coefficients of the continuity equation
    D = NP.zeros(d+1)

    # Dimensionless time inside one control interval
    tau = ca.SX.sym('tau')

    # All collocation time points
    T = NP.zeros((nk,d+1))
    for k in range(nk):
        for j in range(d+1):
            T[k,j] = (k + tau_root[j])


    # For all collocation points
    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        L = 1
        for r in range(d+1):
            if r != j:
                L *= (tau-tau_root[r])/(tau_root[j]-tau_root[r])
        lfcn = ca.SXFunction('lfcn', [tau],[L])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        [D[j]] = lfcn([1.0])

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        tfcn = lfcn.tangent()
        for r in range(d+1):
            C[j][r], _ = tfcn([tau_root[r]])


    # --------------------------------------
    # NLP Variables
    # --------------------------------------

    # Structure holding NLP variables and parameters
    V = ca.struct_symMX([
                      (
                       ca.entry('Xd',repeat=[nk,d+1],struct=xd),
                       ca.entry('XA',repeat=[nk,d],struct=xa),
                       ca.entry('U',repeat=[nk],struct=u),
                       ca.entry('tf'),
                       ca.entry('vlift')
                       )
                      ])


    P = ca.struct_symMX([
                      ca.entry('p', repeat = [nk,d+1], struct = p),
                      ca.entry('toggle_to_energy'),
                    #   ca.entry('tf_previous'),
                      ca.entry('tf')
                      ])

    # --------------------------------------
    # CONSTRAINTS
    # --------------------------------------

    Output_list = {}
    for name in out.keys(): Output_list[name] = []

    # Constraint function for the NLP
    coll_cstr       = [] # Endpoint should match start point
    continuity_cstr = [] # At each collocation point dynamics should meet

    # For all finite elements
    for k in range(nk):

        # For all collocation points
        for j in range(1,d+1):

            # Get an expression for the state derivative at the collocation point
            xp_jk = 0
            for r in range(d+1):
                xp_jk += C[r,j]*V['Xd',k,r]

            # Add collocation equations to the NLP
            [fk] = dynamics([V['Xd',k,j],xp_jk/h/V['tf'], V['XA',k,j-1], V['U',k], P['p',k,j]])
            coll_cstr.append(fk)

        # Get an expression for the state at the end of the finite element
        xf_k = 0
        for r in range(d+1):
            xf_k += D[r]*V['Xd',k,r]

        # Add continuity equation to NLP
        if k < nk-1:
            continuity_cstr.append(V['Xd',k+1,0] - xf_k)

        # Create Outputs
        # NOTE: !!! In principle the range should be range(1,d+1) due to the algebraic variable. But only one output: F_tether is dependent on the alg. var.
        # For plotting F_tether the point on :,0 is wrong and should not be printed. All outputs dependent on algebraic variables are discontinous. !!!

        for j in range(1,d+1):
            [outk] = out_fun([V['Xd',k,j],V['XA',k,j-1],P['p',k,j] ])
            for name in out.keys(): Output_list[name].append(out(outk)[name])

        Output = ca.struct_MX( [ ca.entry(name,expr=Output_list[name]) for name in Output_list.keys() ] )

    # Final time
    xtf = 0
    for r in range(d+1):
        xtf += D[r]*V['Xd',-1,r]

    return V, P, coll_cstr, continuity_cstr, Output
