import numpy as np
from scipy.special import erf

Ti, Ts = 20, -15          # degrees C
alpha = 0.138e-6          # m^2/s
t = 60*86400              # 60 days, in seconds

def f(x):
    return erf(x/(2*np.sqrt(alpha*t))) - 3/7

def fprime(x):
    return (1/np.sqrt(np.pi*alpha*t)) * np.exp(-(x/(2*np.sqrt(alpha*t)))**2)

def newton(f, fprime, x0, tol, maxit=100):
    x = x0
    for n in range(maxit):
        x_new = x - f(x)/fprime(x)
        if abs(x_new - x) < tol:
            return x_new, n+1
        x = x_new
    return x_new, maxit

tol = 1e-13

root1, iter1 = newton(f, fprime, 0.01, tol)
print("starting at x0=0.01:  root =", root1, " iterations =", iter1)

root2, iter2 = newton(f, fprime, 1.0, tol)
print("starting at x0=xbar=1: root =", root2, " iterations =", iter2)