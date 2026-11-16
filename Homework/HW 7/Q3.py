import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / (1 + (10*x)**2)

N = 81   # change this value

j = np.arange(1, N+1)
xn = np.cos((2*j - 1)*np.pi/(2*N))   # Chebyshev nodes
yn = f(xn)

# weights w_j = 1 / prod_{i != j} (x_j - x_i)
w = np.zeros(N)
for k in range(N):
    w[k] = 1 / np.prod(np.delete(xn[k] - xn, k))

# second barycentric formula
xx = np.linspace(-1, 1, 1001)
p = np.zeros(len(xx))
for k in range(len(xx)):
    d = xx[k] - xn
    d[d == 0] = 1e-14            # avoid dividing by zero at a node
    p[k] = np.sum(w*yn/d) / np.sum(w/d)

print("N =", N)
print("max p(x) =", np.max(np.abs(p)))
print("max error =", np.max(np.abs(p - f(xx))))

plt.plot(xn, yn, 'o', label='data')
plt.plot(xx, p, label='p(x)')
plt.plot(xx, f(xx), label='f(x)')
plt.legend()
plt.title(f'Chebyshev nodes, N = {N}')
plt.savefig(f'chebyshev_N{N}.png')