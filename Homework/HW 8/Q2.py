import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator, KroghInterpolator, CubicSpline

def f(x):
    return 1 / (1 + x**2)

def fp(x):
    return -2*x / (1 + x**2)**2

n = 15  # change this value: 5, 10, 15, 20

xn = np.linspace(-5, 5, n + 1)   # n+1 equispaced nodes
xx = np.linspace(-5, 5, 1001)

# (a) Lagrange
lagrange = BarycentricInterpolator(xn, f(xn))(xx)

# (b) Hermite: repeating each node tells Krogh to match f and f' there
xh = np.repeat(xn, 2)
yh = np.ravel(np.column_stack([f(xn), fp(xn)]))
hermite = KroghInterpolator(xh, yh)(xx)

# (c) Natural cubic spline: S'' = 0 at both ends
natural = CubicSpline(xn, f(xn), bc_type='natural')(xx)

# (d) Clamped cubic spline: S' = f' at both ends
clamped = CubicSpline(xn, f(xn), bc_type=((1, fp(-5)), (1, fp(5))))(xx)

print("n =", n)
print("max error, Lagrange:       ", np.max(np.abs(lagrange - f(xx))))
print("max error, Hermite:        ", np.max(np.abs(hermite - f(xx))))
print("max error, natural spline: ", np.max(np.abs(natural - f(xx))))
print("max error, clamped spline: ", np.max(np.abs(clamped - f(xx))))

plt.plot(xx, f(xx), 'k', linewidth=2, label='f(x)')
plt.plot(xx, lagrange, label='Lagrange')
plt.plot(xx, hermite, label='Hermite')
plt.plot(xx, natural, label='Natural spline')
plt.plot(xx, clamped, '--', label='Clamped spline')
plt.plot(xn, f(xn), 'o', label='nodes')
plt.ylim(-1, 2)
plt.legend()
plt.title(f'n = {n}')
plt.savefig(f'interp_n{n}.png')