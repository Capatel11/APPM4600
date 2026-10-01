import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def f(x):
    return x**6 - x - 1

def fp(x):
    return 6*x**5 - 1

alpha = brentq(f, 1, 2)   # exact largest root

def newton(x0, n):
    xs = [x0]
    for _ in range(n):
        x = xs[-1]
        xs.append(x - f(x)/fp(x))
    return xs

def secant(x0, x1, n):
    xs = [x0, x1]
    for _ in range(n):
        xk, xk1 = xs[-2], xs[-1]
        xs.append(xk1 - f(xk1)*(xk1-xk)/(f(xk1)-f(xk)))
    return xs

xs_newton = newton(2.0, 6)
xs_secant = secant(2.0, 1.0, 9)

err_newton = [abs(x - alpha) for x in xs_newton]
err_secant = [abs(x - alpha) for x in xs_secant]

print("Newton errors:", err_newton)
print("Secant errors:", err_secant)

plt.loglog(err_newton[:-1], err_newton[1:], 'o-', label='Newton')
plt.loglog(err_secant[:-1], err_secant[1:], 's-', label='Secant')
plt.xlabel('|x_k - alpha|')
plt.ylabel('|x_(k+1) - alpha|')
plt.legend()
plt.show()