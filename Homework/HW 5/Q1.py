import numpy as np

def F(v):
    x, y = v
    return np.array([3*x**2 - y**2,
                     3*x*y**2 - x**3 - 1])

def J(v):
    x, y = v
    return np.array([[6*x, -2*y],
                     [3*y**2 - 3*x**2, 6*x*y]])

tol = 1e-12
maxit = 100

# a) fixed matrix iteration
A = np.array([[1/6, 1/18],
              [0,   1/6]])

v = np.array([1.0, 1.0])
for n in range(maxit):
    v_new = v - A @ F(v)
    if np.linalg.norm(v_new - v) < tol:
        break
    v = v_new
print("a) fixed matrix: solution =", v_new, "iterations =", n+1)

# c) Newton's method
v = np.array([1.0, 1.0])
for n in range(maxit):
    v_new = v - np.linalg.solve(J(v), F(v))
    if np.linalg.norm(v_new - v) < tol:
        break
    v = v_new
print("c) Newton's method: solution =", v_new, "iterations =", n+1)