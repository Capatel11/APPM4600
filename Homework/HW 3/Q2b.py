import numpy as np
from scipy.special import erf

Ti, Ts = 20, -15          # degrees C
alpha = 0.138e-6          # m^2/s
t = 60*86400              # 60 days, in seconds

def f(x):
    return erf(x/(2*np.sqrt(alpha*t))) - 3/7

def bisection(f, a, b, tol):
    fa = f(a)
    count = 0
    while (b - a)/2 > tol:
        c = (a + b)/2
        fc = f(c)
        if fa*fc < 0:
            b = c
        else:
            a = c
            fa = fc
        count += 1
    return (a + b)/2, count

a0, b0 = 0, 1
tol = 1e-13

root, iterations = bisection(f, a0, b0, tol)

print("depth (bisection):", root)
print("iterations:", iterations)