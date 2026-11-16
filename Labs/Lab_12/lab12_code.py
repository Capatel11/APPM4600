"""
 This script provides you the opprotunity play with different linear algebra solvers.
You will learn about the performance of the different solvers and how to build
test some test matrices to challenge the different methods. 
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
import numpy.linalg as la
import scipy.linalg as scipy
import time

np.seterr(divide='ignore', over='ignore', invalid='ignore')

def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     N = 100
 
     ''' Right hand side'''
     b = np.random.rand(N,1)
     A = np.random.rand(N,N)
  
     x = scipy.solve(A,b)
     
     test = np.matmul(A,x)
     r = la.norm(test-b)
     
     print(r)

     # 3.2.1
     lu, piv = scipy.lu_factor(A)
     x_lu = scipy.lu_solve((lu, piv), b)
     test_lu = np.matmul(A,x_lu)
     print(f"LU_factor: {la.norm(test_lu-b)}")


     # 3.2.2
     for N in [1,100,500,1000,2000,4000,5000]:
          A = np.random.rand(N,N)
          b = np.random.rand(N,1)

          t0 = time.time()
          x = scipy.solve(A,b)
          t_solve = time.time() - t0

          t0 = time.time()
          lu, piv = scipy.lu_factor(A)
          t_factor = time.time() - t0


          t0 = time.time()
          x_lu = scipy.lu_solve((lu, piv),b)
          t_lusolve = time.time() - t0

          print(f"With N={N}: t_solve = {t_solve}, t_factor = {t_factor}, t_lusolve = {t_lusolve}, LU = {t_factor + t_lusolve}")




     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)


     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
