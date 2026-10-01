import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

Ti, Ts = 20, -15          # degrees C
alpha = 0.138e-6          # m^2/s
t = 60*86400              # 60 days, in seconds

def f(x):
    return erf(x/(2*np.sqrt(alpha*t))) - 3/7

xbar = 1.0   # meters; f(xbar) > 0 here

x = np.linspace(0, xbar, 200)
y = f(x)

plt.plot(x, y)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('depth x (m)')
plt.ylabel('f(x)')
plt.show()

print("f(0) =", f(0))
print("f(xbar) =", f(xbar))