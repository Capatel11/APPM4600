import numpy as np

def g(x):
    return -np.sin(2*x) + 5*x/4 - 3/4

def gprime(x):
    return -2*np.cos(2*x) + 5/4

def fixed_point(x0, tol, maxit=500):
    x = x0
    for n in range(maxit):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new, n+1, True
        x = x_new
    return x, maxit, False

# the 5 roots found in part (a)
roots = [-0.898356581546, -0.544442400681, 1.732069502144, 3.161826486552, 4.517789514180]

tol = 0.5e-10  # gives 10 correct digits

for r in roots:
    x0 = r + 0.01   # start close to each known root
    result, iterations, converged = fixed_point(x0, tol)
    print(f"root near {r:.6f}: converged={converged}, result={result:.10f}")