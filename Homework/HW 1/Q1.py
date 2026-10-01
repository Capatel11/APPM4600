import matplotlib.pyplot as plt
import numpy as np
import math


p = lambda x: x**9 - 18*(x**8) + 144*(x**7) - 672*(x**6) + 2016*(x**5) - 4032*(x**4) + 5376*(x**3) - 4608*(x**2) + 2304*x - 512
p2 = lambda x: (x-2)**9

x = np.arange(1.920, 2.081, 0.001)

plt.plot(x, p2(x))
plt.show()