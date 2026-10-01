import numpy as np
import matplotlib.pyplot as plt

def direct(x, delta):
    return np.cos(x + delta) - np.cos(x)

def rewritten(x, delta):
    return -2 * np.sin(x + delta / 2) * np.sin(delta / 2)

deltas = np.array([10.0 ** k for k in range(-16, 1)])  # 1e-16 ... 1e0

x_values = [np.pi, 1e6]
labels = ['x = pi', 'x = 1e6']

plt.figure()
for x_val, label in zip(x_values, labels):
    diff = np.abs(direct(x_val, deltas) - rewritten(x_val, deltas))
    plt.loglog(deltas, diff, 'o-', label=label)

plt.xlabel('Delta')
plt.ylabel('Rewritten Expression: cos(x+delta)-cos(x)')
plt.legend()
plt.show()