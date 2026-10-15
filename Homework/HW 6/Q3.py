import numpy as np

def F(v):
    x, y, z = v
    return np.array([x + np.cos(x*y*z) - 1,
                     (1 - x)**0.25 + y + 0.05*z**2 - 0.15*z - 1,
                     -x**2 - 0.1*y**2 + 0.01*y + z - 1])

def J(v):
    x, y, z = v
    s = np.sin(x*y*z)
    return np.array([[1 - y*z*s,            -x*z*s,      -x*y*s],
                     [-0.25*(1 - x)**-0.75,  1,           0.1*z - 0.15],
                     [-2*x,                 -0.2*y + 0.01, 1]])

def g(v):                       # function to minimize: sum of squares
    return np.sum(F(v)**2)

def newton(v, tol, maxit=100):
    for n in range(maxit):
        v_new = v - np.linalg.solve(J(v), F(v))
        if np.linalg.norm(v_new - v) < tol:
            return v_new, n+1
        v = v_new
    return v, maxit

def steepest_descent(v, tol, maxit=10000):
    for n in range(maxit):
        g1 = g(v)
        grad = 2 * J(v).T @ F(v)
        norm = np.linalg.norm(grad)
        if norm == 0:
            return v, n
        d = grad / norm

        # find a step a3 that decreases g
        a3 = 1.0
        g3 = g(v - a3*d)
        while g3 >= g1:
            a3 = a3/2
            g3 = g(v - a3*d)
            if a3 < tol/2:
                return v, n

        # quadratic interpolation through a = 0, a2, a3
        a2 = a3/2
        g2 = g(v - a2*d)
        h1 = (g2 - g1)/a2
        h2 = (g3 - g2)/(a3 - a2)
        h3 = (h2 - h1)/a3
        a0 = 0.5*(a2 - h1/h3) if h3 != 0 else a3
        g0 = g(v - a0*d)

        a = a0 if g0 < g3 else a3
        v_new = v - a*d
        if np.linalg.norm(v_new - v) < tol:
            return v_new, n+1
        v = v_new
    return v, maxit

x0 = np.array([0.0, 0.0, 0.0])

v, n = newton(x0, 1e-6)
print("Newton:                  ", v, " iterations =", n)

v, n = steepest_descent(x0, 1e-6)
print("Steepest descent:        ", v, " iterations =", n)

v_sd, n_sd = steepest_descent(x0, 5e-2)
v, n = newton(v_sd, 1e-6)
print("Steepest descent (5e-2): ", v_sd, " iterations =", n_sd)
print("  then Newton:           ", v, " iterations =", n)