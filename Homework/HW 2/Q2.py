import numpy as np

A = 0.5 * np.array([[1, 1],
                     [1 + 1e-10, 1 - 1e-10]])

print(np.linalg.cond(A))