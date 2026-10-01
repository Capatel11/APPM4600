import numpy as np
import matplotlib.pyplot as plt

def rewritten(x, delta):
    return -2 * np.sin(x + delta / 2) * np.sin(delta / 2)

def taylor(x, delta):
    return -delta * np.sin(x) - (delta**2 / 2) * np.cos(x)

deltas = np.array([10.0 ** k for k in range(-16, 1)])  # 1e-16 ... 1e0

x_values = [np.pi, 1e6]
labels = ['x = pi', 'x = 1e6']

plt.figure()
for x_val, label in zip(x_values, labels):
    diff = np.abs(rewritten(x_val, deltas) - taylor(x_val, deltas))
    plt.loglog(deltas, diff, 'o-', label=label)

plt.xlabel('delta')
plt.ylabel('|rewritten expression - taylor approximation|')
plt.legend()
plt.show()