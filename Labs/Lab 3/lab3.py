import numpy as np

def fixedpt(f, x0, tol, max):
    x = np.zeros(max + 1)
    x[0] = x0
    for i in range(max):
        x1 = f(x[i])
        x[i + 1] = x1
        if abs(x1 - x[i]) < tol:
            return x[:i + 2], i
    return x, max


f = lambda x: (10/(x+4))**(1/2)
tol = 1e-10
xs, i = fixedpt(f, 1.0, tol, 100)
print(xs)
print(i)

def epsilon()

psol = 1.3652300134140976
it = 10
alpha = np.log(ep[it + 1]/ep[it])/np.log(ep[it]/ep[it - 1])
print(alpha)
print(xs[-1],xs[-2],xs[-3])


def Aitkens(f, p, tol, max):
    A = np.zeros(max + 1)
    A[0] = p[0]
    for i in range(max):
        A[i+1] = p[i] - ((p[i+1] - p[i])**2)/(p[i+2] - 2*p[i+1] + p[i])
        if abs(A[i] - A[i+1]) < tol:
            return A[:i + 2], i
    return A, max


#pn, i = Aitkens(f, xs, tol, i)
#print(pn)
#print(i)