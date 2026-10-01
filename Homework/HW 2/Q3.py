from mpmath import mp, mpf, exp, log10, fabs
import math

mp.dps = 60

x = mpf("9.999999995000000e-10")
exact = exp(x) - 1

alg = math.exp(float(x)) - 1
linear = x
quadratic = x + x**2/2

digits_alg = float(-log10(fabs((mpf(alg) - exact) / exact)))
digits_linear = float(-log10(fabs((linear - exact) / exact)))
digits_quadratic = float(-log10(fabs((quadratic - exact) / exact)))

print("alg:", digits_alg)
print("Taylor, x only:           ", digits_linear)
print("Taylor, x + x^2/2:        ", digits_quadratic)