import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x - 4*np.sin(2*x) - 3

# all roots satisfy x = 4 sin(2x) + 3, and |4 sin(2x)| <= 4, so every root lies in [-1, 7]
x = np.linspace(-2, 8, 2000)
y = f(x)

# count sign changes (each one brackets a root)
num_roots = np.sum(np.diff(np.sign(y)) != 0)

plt.plot(x, y)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
# plt.show()

print("number of zero crossings:", num_roots)