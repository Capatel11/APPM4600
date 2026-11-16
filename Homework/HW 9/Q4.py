import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 1001)
f = np.sin(x)

maclaurin = x - x**3/6 + x**5/120
pade_33 = (x - 7*x**3/60) / (1 + x**2/20)        # (a) M = N = 3
pade_42 = (x - 7*x**3/60) / (1 + x**2/20)        # (b) M = 4, N = 2 (same as (a))
pade_24 = x / (1 + x**2/6 + 7*x**4/360)          # (c) M = 2, N = 4

print("max error on [0,5]:")
print("  Maclaurin:   ", np.max(np.abs(f - maclaurin)))
print("  Pade (3,3):  ", np.max(np.abs(f - pade_33)))
print("  Pade (4,2):  ", np.max(np.abs(f - pade_42)))
print("  Pade (2,4):  ", np.max(np.abs(f - pade_24)))

plt.semilogy(x, np.abs(f - maclaurin), label='Maclaurin (degree 6)')
plt.semilogy(x, np.abs(f - pade_33), label='Pade M=N=3 (and M=4, N=2)')
plt.semilogy(x, np.abs(f - pade_24), '--', label='Pade M=2, N=4')
plt.xlabel('x')
plt.ylabel('|error|')
plt.legend()
plt.savefig('pade_error.png')
print("plot saved as pade_error.png")