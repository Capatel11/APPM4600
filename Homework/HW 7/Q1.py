import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / (1 + (10*x)**2)

def eval_poly(c, x):
    # Horner's method, c = [c1, ..., cn] with c1 multiplying x^(n-1)
    p = np.zeros_like(x)
    for ck in c:
        p = p*x + ck
    return p

N = 2   # change this value

h = 2/(N - 1)
x = -1 + np.arange(N)*h          # x_i = -1 + (i-1)h, i = 1..N
y = f(x)

# Vandermonde matrix: row j = [x_j^(N-1), x_j^(N-2), ..., x_j, 1]
V = np.array([[xj**(N-1-k) for k in range(N)] for xj in x])
c = np.linalg.inv(V) @ y

xx = np.linspace(-1, 1, 1001)
p = eval_poly(c, xx)

plt.plot(x, y, 'o', label='data')
plt.plot(xx, p, label='p(x)')
plt.plot(xx, f(xx), label='f(x)')
plt.legend()
plt.title(f'N = {N}')
plt.show()

print("N =", N)
print("max p(x) =", np.max(p))