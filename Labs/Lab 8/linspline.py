"""
 This script provides you the opprotunity to insert your own line evaluator.
 You will practice some python skills while getting a better understanding of
 how splines work.

 COMPLETED VERSION.  Changes from the template:
   * added the pre-lab line evaluator  eval_line(x0,f0,x1,f1,alpha)
   * f uses np.exp (vectorized) instead of math.exp so f(xeval) works on arrays
   * driver now passes Neval to eval_lin_spline (template call was missing it)
   * xint(jint) -> xint[jint]  (Python indexes arrays with square brackets)
   * fixed the indentation of the a1/fa1/b1/fb1 block
   * filled in the index search (ind, n) and the loop over kk
   * plt.legend -> plt.legend()
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


import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv


def driver():

    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1

    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)

    ''' number of intervals'''
    Nint = 10

    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)

    ''' evaluate f at the evaluation points'''
    fex = f(xeval)

    print('max abs error = ', np.max(np.abs(yeval-fex)))

    plt.figure()
    plt.plot(xeval,fex,'ro-',label='exact function')
    plt.plot(xeval,yeval,'bs--',label='linear spline')
    plt.xlabel('x')
    plt.legend()
    plt.show()

    err = abs(yeval-fex)
    plt.figure()
    plt.semilogy(xeval,err,'ro--',label='absolute error')
    plt.legend()
    plt.show()


def eval_line(x0,f0,x1,f1,alpha):
    '''PRE-LAB: evaluate the line through (x0,f0) and (x1,f1) at alpha.
       Slope m = (f1-f0)/(x1-x0); the line is  f0 + m*(alpha-x0).
       Works for a scalar alpha or a numpy array of points.'''
    m = (f1-f0)/(x1-x0)
    return f0 + m*(alpha-x0)


def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)

    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval)

    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        ind = np.where((xeval >= xint[jint]) & (xeval <= xint[jint+1]))[0]
        '''let n denote the length of ind'''
        n = len(ind)

        '''temporarily store your info for creating a line in the interval of
         interest'''
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)

        for kk in range(n):
            '''use your line evaluator to evaluate the lines at each of the points
            in the interval'''
            yeval[ind[kk]] = eval_line(a1,fa1,b1,fb1,xeval[ind[kk]])

    return yeval


if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()