import numpy as np

def f(p):
    x, y, z = p
    return x**2 + 4*y**2 + 4*z**2 - 16

def grad_f(p):
    x, y, z = p
    return np.array([2*x, 8*y, 8*z])

p = np.array([1.0, 1.0, 1.0])
tol = 1e-14
points = [p]

for n in range(50):
    g = grad_f(p)
    d = f(p) / np.dot(g, g)
    p_new = p - d*g
    points.append(p_new)
    if np.linalg.norm(p_new - p) < tol:
        break
    p = p_new

p_final = points[-1]
print("point on ellipsoid:", p_final)
print("f at that point:", f(p_final))
print("iterations:", n+1)

# errors relative to the final point, and ratio e_{k+1}/e_k^2
errors = [np.linalg.norm(q - p_final) for q in points[:-1]]
print("\nk, error, error_(k+1)/error_k^2")
for k in range(len(errors)-1):
    ratio = errors[k+1]/errors[k]**2 if errors[k+1] > 0 else 0
    print(k, errors[k], ratio)