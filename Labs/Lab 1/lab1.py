import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 21)
y = np.arange(0, 10, .5)

# print(f"The first 3 entries are {x[0]}, {x[1]}, {x[2]}")


w = 10**np.linspace(1, 10, 10)
x = np.linspace(1, 10, 10)
s = 3*w

plt.semilogy(x,w)
plt.semilogy(x,s)
plt.show()

