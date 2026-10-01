"""
 This compares Newton's method, Broyden and Lazy Newton for 
 computing the roots of vector valued functions.
 The function and the Jacobian are stored in subroutines and need 
 to be changed for different problems.  
 
"""

############################################# 
"""
Copyright (C) 2025  Adrianna M. Gillman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
############################################# 


import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver():

    # initial guesses from the lab, plus one of my own that converges
    guesses = [np.array([1, 1])]
#  np.array([2.0, 0.5]), np.array([3.0, 5.0]), np.array([1.5, 1.5])

    Nmax = 100000
    tol = 1e-10

    for x0 in guesses:
        print('==== initial guess x0 =', x0, '====')
        runMethod('Newton', Newton, x0, tol, Nmax, 50)
        runMethod('Lazy Newton', LazyNewton, x0, tol, Nmax, 20)
        runMethod('Slacker Newton', SlackerNewton, x0, tol, Nmax, 20)
        # runMethod('Broyden', Broyden, x0, tol, Nmax, 20)
        print()

def runMethod(name, method, x0, tol, Nmax, nruns):
# times a method and prints its results; reports a failure instead of crashing

    try:
        t = time.time()
        for j in range(nruns):
          [xstar,ier,its] = method(x0,tol,Nmax)
        elapsed = time.time()-t
    except (np.linalg.LinAlgError, OverflowError) as e:
        print(name + ': failed with:', e)
        return

    if not np.all(np.isfinite(xstar)):
        ier = 1
    print(xstar)
    print(name + ': the error message reads:',ier)
    print(name + ': took this many seconds:',elapsed/nruns)
    print(name + ': number of iterations is:',its)
    print(name + ': norm of F at xstar is:',norm(evalF(xstar)))

def evalF(x): 
# vector function that you want to find the roots of

    F = np.zeros(2)
    
    # F[0] = 4*(x[0]**2) + x[1]**2 - 4
    # F[1] = x[0] + x[1] - np.sin(x[0]-x[1])

    F[0] = np.exp(10*x[0]) + x[1] - 1
    F[1] = 10*x[0]**2 + x[1]
 
    return F
    
def evalJ(x): 
# Jacobian of the vector function you want to find the roots of
    
    # J = np.array([[8.*x[0], 2.*x[1]], 
    #         [1 - np.cos(x[0]-x[1]), 1 + np.cos(x[0]-x[1])]])
    
    J = np.array([[10*np.exp(10*x[0]), 1], 
       [20*x[0], 1]])

    return J


def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]
           
def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   
    
def SlackerNewton(x0,tol,Nmax):

    ''' Slacker Newton = keep the old inverse Jacobian until a test says it
        has gone stale, then recompute it at the current iterate'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    xJ = x0              # where the Jacobian was last computed
    sprev = np.inf       # norm of the previous step
    Fprev = np.inf       # norm of F at the previous iterate

    for its in range(Nmax):

        F = evalF(x0)
        s = -Jinv.dot(F)

        ## the step got bigger than the last one (iterates not closing in)
        recompute = norm(s) > sprev

        ## fixed schedule, recompute every 3rd iteration
        # recompute = (its > 0) and (its % 3 == 0)

        if recompute:
           J = evalJ(x0)
           Jinv = inv(J)
           xJ = x0
           s = -Jinv.dot(F)

        x1 = x0 + s

        if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]

        sprev = norm(s)
        Fprev = norm(F)
        x0 = x1

    xstar = x1
    ier = 1
    return[xstar,ier,its]

def Broyden(x0,tol,Nmax):
    '''tol = desired accuracy
    Nmax = max number of iterations'''

    '''Sherman-Morrison 
   (A+xy^T)^{-1} = A^{-1}-1/p*(A^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''

    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''


    '''In Broyden 
    x = [F(xk)-F(xk-1)-\hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''

    ''' implemented as in equation (10.16) on page 650 of text'''
    
    '''initialize with 1 newton step'''
    
    A0 = evalJ(x0)

    v = evalF(x0)
    A = np.linalg.inv(A0)

    s = -A.dot(v)
    xk = x0+s
    for  its in range(Nmax):
       '''(save v from previous step)'''
       w = v
       ''' create new v'''
       v = evalF(xk)
       '''y_k = F(xk)-F(xk-1)'''
       y = v-w;                   
       '''-A_{k-1}^{-1}y_k'''
       z = -A.dot(y)
       ''' p = s_k^tA_{k-1}^{-1}y_k'''
       p = -np.dot(s,z)                 
       u = np.dot(s,A) 
       ''' A = A_k^{-1} via Morrison formula'''
       tmp = s+z
       tmp2 = np.outer(tmp,u)
       A = A+1./p*tmp2
       ''' -A_k^{-1}F(x_k)'''
       s = -A.dot(v)
       xk = xk+s
       if (norm(s)<tol):
          alpha = xk
          ier = 0
          return[alpha,ier,its]
    alpha = xk
    ier = 1
    return[alpha,ier,its]
     
        
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()       
