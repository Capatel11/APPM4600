from mpmath import mp, mpf, exp, diff, findroot, fabs

mp.dps = 60   # working precision (decimal digits)

def f(x):
    return exp(3*x) - 27*x**6 + 27*x**4*exp(x) - 9*x**2*exp(2*x)

def fp(x):
    return diff(f, x)

def fpp(x):
    return diff(f, x, 2)

m = 3   # known multiplicity of the root (f = (e^x - 3x^2)^3)

# reference root, found via the simple factor e^x - 3x^2 = 0
def h(x):
    return exp(x) - 3*x**2

alpha = findroot(h, mpf(4))

def newton_step(x):
    return x - f(x)/fp(x)

def modified_class_step(x):
    return x - f(x)*fp(x)/(fp(x)**2 - f(x)*fpp(x))

def modified_m_step(x):
    return x - m*f(x)/fp(x)

def run(step, x0, tol=mpf('1e-45'), maxit=200):
    x = x0
    for n in range(maxit):
        x_new = step(x)
        if fabs(x_new - x) < tol:
            return x_new, n+1
        x = x_new
    return x_new, maxit

x0 = mpf(4)

print("reference root alpha =", alpha)

for name, step in [("Newton", newton_step),
                    ("Modified (class)", modified_class_step),
                    ("Modified (m known)", modified_m_step)]:
    root, iterations = run(step, x0)
    print(f"\n{name}:")
    print("  root       =", root)
    print("  iterations =", iterations)