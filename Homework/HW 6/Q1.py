import numpy as np

def F(v):
    x, y = v
    return np.array([3*x**2 + 4*y**2 - 1,
                     y**3 - 8*x**3 - 1])

def J(v):
    x, y = v
    return np.array([[6*x,       8*y],
                     [-24*x**2,  3*y**2]])

A = np.array([[0.016, -0.17],
              [0.52,  -0.26]])

tol = 1e-4
maxit = 100
guesses = [(-0.5, 0.25), (-1.5, 1.25), (-3, 3)]

def fixed_point(v):
    for n in range(maxit):
        v_new = v - A @ F(v)
        if not np.all(np.isfinite(v_new)) or np.linalg.norm(v_new) > 1e10:
            return v_new, n+1, "diverged"
        if np.linalg.norm(v_new - v) < tol:
            return v_new, n+1, "converged"
        v = v_new
    return v, maxit, "did not converge"

def newton(v):
    for n in range(maxit):
        v_new = v - np.linalg.solve(J(v), F(v))
        if np.linalg.norm(v_new - v) < tol:
            return v_new, n+1, "converged"
        v = v_new
    return v, maxit, "did not converge"

print("(a) Fixed point:")
for g in guesses:
    v, n, status = fixed_point(np.array(g, dtype=float))
    print(f"  start {g}: {status}, iterations = {n}, result = {v}")

print("\n(b) Newton:")
for g in guesses:
    v, n, status = newton(np.array(g, dtype=float))
    print(f"  start {g}: {status}, iterations = {n}, result = {v}")