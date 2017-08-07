'''
collecting def
Python Version 2.7 / Casadi version 2.4.1
- Author: Elena Malz, Chalmers 2016
'''
from casadi import *
from casadi.tools import *
import numpy as np


def Lagrange_poly(x,y):
    "creates an lagrange polynomial through each data point based on x and y data"
    t = SX.sym('t')
    d = x.shape[0]                          # amount of parameters

    poly = 0
    for j in range(d):                  # for all data points ...
        L = y[j]                        # parameter = fct output
        for r in range(d):
            if r != j:
                L *= (t-x[r])/(x[j]-x[r])
        poly+=L
    lfcn = SXFunction('lfcn', [t],[poly])
    return lfcn

def smooth_Lagrange_poly(x,y):
    t    = SX.sym('t')
    d    = len(x)                       # amount of parameters
    tau  = SX.sym('tau',d)              # parameter as minimisation variable
    poly = 0

    for j in range(d):                  # for all data points ...
        L = tau[j]
        for r in range(d):
            if r != j:
                L *= (t-x[r])/(x[j]-x[r])
        poly+=L
    L_fun   = SXFunction('L_fun', [t,tau],[poly])
    ddL_fun = L_fun.hessian(0)          # second order derivative to
    [ddL,_,_]  = ddL_fun([t,tau])

    # minimise tau = fct output, incl penalize curvature
    res = 0.1 *  sum([(L_fun([x[k],tau]) - y[k])**2 for k in range(d)])[0]
    res += sum([ddL_fun([x[k],tau])[0]**2 * 1e4 for k in range(d)])[0]

    Cost= SXFunction('cost',[tau],[res])
    nlp = SXFunction('nlp', nlpIn(x=tau),nlpOut(f=res))
    solver = NlpSolver("solver", "ipopt", nlp)
    sol = solver({})
    tau_opt = sol['x']                  # optimal parameter for polynomial
    return  L_fun, tau_opt


def create_LSpoly(d,x,y):
    "creates polynomial by least squares fitting of order d. Builds Vandermonde matrix manually"
    # x = np.append(x,0)              # add (0,0) as data point
    # y = np.append(y,0)

    a = d+1                         # number of parameters including a0
    M = SX.sym('M',2*d+1)           # all the exponents from 1 to 2k
    sizex =  x.shape[0]             # number of data points x
    M[0] = sizex                    # first entry in matrix

    for k in range(1, M.shape[0]): # collect all matrix entries
        M[k] = sum([x[o]**k for o in range(0,sizex)])

    sumM = SX(a,a)                  # create actual matrix
    for j in range(0,a):
            for k in range(0,a):
                sumM[j,k] = M[k+j]

    B = SX(a,1)                     # create B vector (sum of y)
    for k in range(0,a):
        B[k] = sum([y[o]*x[o]**k for o in range(0,sizex)])

    X = np.linalg.solve(sumM, B)    # parameters order: low to high power
    xvar = SX.sym('xvar')
    poly = X[0]
    for k in range(1,X.shape[0]):
        poly += X[k]*xvar**(k)      # create polynomial
    pfun = SXFunction('poly',[xvar],[poly])
    return pfun, X, poly
