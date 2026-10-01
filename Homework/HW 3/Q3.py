import numpy as np

def g_a(x): return x*(1 + (7 - x**5)/x**2)**3
def g_b(x): return x - (x**5 - 7)/x**2
def g_c(x): return x - (x**5 - 7)/(5*x**4)
def g_d(x): return x - (x**5 - 7)/12

choice = 'd'   # functions: a, b, c, d

funcs = {'a': g_a, 'b': g_b, 'c': g_c, 'd': g_d}
g = funcs[choice]

x0 = 1.0
tol = 1e-10
maxit = 2000

x = x0
converged = False
for n in range(maxit):
    x_new = g(x)
    if abs(x_new - x) < tol:
        converged = True
        break
    if abs(x_new) > 1e15:
        break
    x = x_new

print("function:", choice)
print("converged:", converged)
print("result:", x_new)
print("iterations:", n+1)