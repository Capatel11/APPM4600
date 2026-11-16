import numpy as np
from scipy.integrate import quad

def f(s):
    return 1 / (1 + s**2)

a, b = -5, 5
I_exact = 2*np.arctan(5)

def trapezoid(n):
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return h * (f(x[0])/2 + np.sum(f(x[1:-1])) + f(x[-1])/2)

def simpson(n):
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return h/3 * (f(x[0]) + 4*np.sum(f(x[1:-1:2])) + 2*np.sum(f(x[2:-1:2])) + f(x[-1]))

n_T = 1291   # trapezoidal error bound
n_S = 108    # simpson error bound

T = trapezoid(n_T)
S = simpson(n_S)
print(f"Trapezoidal: n = {n_T}, T_n = {T:.10f}, error = {abs(T - I_exact):.3e}, f evaluations = {n_T + 1}")
print(f"Simpson:     n = {n_S}, S_n = {S:.10f}, error = {abs(S - I_exact):.3e}, f evaluations = {n_S + 1}")

for tol in [1e-6, 1e-4]:
    val, err, info = quad(f, a, b, epsabs=tol, epsrel=tol, full_output=1)
    print(f"quad (tol = {tol:.0e}): value = {val:.10f}, error = {abs(val - I_exact):.3e}, f evaluations = {info['neval']}")

