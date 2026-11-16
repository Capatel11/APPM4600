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

     # 3.2 Exercise 1: solve the same system with an LU factorization
     lu, piv = scipy.lu_factor(A)             # build PA = LU
     x_lu = scipy.lu_solve((lu,piv),b)        # two triangular solves
     print('LU residual:', la.norm(np.matmul(A,x_lu)-b))

     # 3.2 Exercise 2: time both techniques for each N
     for N in [100, 500, 1000, 2000, 4000, 5000]:
          A = np.random.rand(N,N)
          b = np.random.rand(N,1)

          t0 = time.time()
          x = scipy.solve(A,b)
          t_solve = time.time() - t0

          t0 = time.time()
          lu, piv = scipy.lu_factor(A)          # (i) build the factorization
          t_fact = time.time() - t0

          t0 = time.time()
          x_lu = scipy.lu_solve((lu,piv),b)     # (ii) one solve with it
          t_lusolve = time.time() - t0

          print('N =', N, ' solve:', t_solve, ' LU factor:', t_fact, ' LU solve:', t_lusolve)

     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)

     # 3.4 Exercises 1 and 2: smallest entry of d is 1e-10 (as given)
     print('smallest d = 1e-10')
     solve_rect(A)

     # 3.4 Exercise 3: make the smallest entry of d smaller than 1e-10
     for p in [12, 14, 16]:
          print('smallest d = 1e-%d' % p)
          solve_rect(create_rect(N,M,p))


def solve_rect(A):
     ''' solves Ax = b with the normal equation and with QR.
         To validate, we make b = A x_true so we know the right answer.'''
     N, M = A.shape
     x_true = np.random.rand(M,1)
     b = np.matmul(A,x_true)

     # normal equation:  (A^T A) x = A^T b
     x_normal = scipy.solve(np.matmul(A.T,A), np.matmul(A.T,b))

     # QR:  A = QR,  so  R x = Q^T b  (back substitution)
     Q, R = la.qr(A)
     x_qr = scipy.solve_triangular(R, np.matmul(Q.T,b))

     print('   normal eq error:', la.norm(x_normal-x_true)/la.norm(x_true))
     print('   QR error:       ', la.norm(x_qr-x_true)/la.norm(x_true))

     
def create_rect(N,M,p=10):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     # p = 10 gives the original matrix; smallest entry of d is 10^(-p)
     a = np.linspace(1,p,M)
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