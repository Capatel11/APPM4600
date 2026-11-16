import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / (1 + (10*x)**2)

N = 81   # change this value

xn = np.linspace(-1, 1, N)       # interpolation nodes
yn = f(xn)

# weights w_j = 1 / prod_{i != j} (x_j - x_i)
w = np.zeros(N)
for j in range(N):
    w[j] = 1 / np.prod(np.delete(xn[j] - xn, j))

# second barycentric formula
xx = np.linspace(-1, 1, 1001)
p = np.zeros(len(xx))
for k in range(len(xx)):
    d = xx[k] - xn
    d[d == 0] = 1e-14            # avoid dividing by zero at a node
    p[k] = np.sum(w*yn/d) / np.sum(w/d)

print("N =", N)
print("max p(x) =", np.max(np.abs(p)))

plt.plot(xn, yn, 'o', label='data')
plt.plot(xx, p, label='p(x)')
plt.plot(xx, f(xx), label='f(x)')
plt.legend()
plt.title(f'N = {N}')
plt.savefig(f'barycentric_N{N}.png')

plt.xlim(-0.2, 0.2)              # zoom in near x = 0
plt.ylim(0, 1.1)
plt.savefig(f'barycentric_N{N}_zoom.png')