import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(x) * np.cos(x)

a, b = 0, np.pi
I_true = -12.070346316389632
tol = 1e-10

def trapezoid(n):
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return h * (f(x[0])/2 + np.sum(f(x[1:-1])) + f(x[-1])/2)

def simpson(n):            # n must be even
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return h/3 * (f(x[0]) + 4*np.sum(f(x[1:-1:2])) + 2*np.sum(f(x[2:-1:2])) + f(x[-1]))

for name, rule in [("Trapezoidal", trapezoid), ("Simpson", simpson)]:
    n = 2
    hs, errs = [], []
    while True:
        h = (b - a) / n
        err = abs(rule(n) - I_true)
        hs.append(h)
        errs.append(err)
        if err / abs(I_true) < tol:
            break
        n = 2 * n
    slope = np.polyfit(np.log(hs[-5:]), np.log(errs[-5:]), 1)[0]
    print(f"{name}: n = {n}, h = {h:.3e}, approx = {rule(n):.12f}, slope = {slope:.3f}")
    plt.loglog(hs, errs, 'o-', label=name)

plt.xlabel('h')
plt.ylabel('absolute error')
plt.legend()
plt.savefig('trap_simpson_error.png')
print("plot saved as trap_simpson_error.png")