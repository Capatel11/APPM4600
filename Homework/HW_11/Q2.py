import matplotlib.pyplot as plt
import numpy as np


def driver():

    f = lambda x: np.sin(x)
    a = 0
    b = np.pi

    # exact integral
    I_ex = 2

    nrefine = 5

    # Romberg-Trapezoidal: panels 1, 2, 4, ..., 2^nrefine
    n_trap = 2**np.arange(0, nrefine+1)
    RomT = rombergTrap(a, b, nrefine, f)
    errT = abs(I_ex - np.diag(RomT))

    # Romberg-Simpson: panels 2, 4, 8, ..., 2^(nrefine+1)  (Simpson needs n even)
    n_simp = 2**np.arange(1, nrefine+2)
    RomS = rombergSimp(a, b, nrefine, f)
    errS = abs(I_ex - np.diag(RomS))

    print('Romberg-Trapezoidal diagonal errors:', errT)
    print('Romberg-Simpson diagonal errors:    ', errS)

    plt.semilogy(n_trap, errT, 'o-', label='Romberg-Trapezoidal')
    plt.semilogy(n_simp, errS, 's-', label='Romberg-Simpson')
    plt.xlabel('n (number of panels)')
    plt.ylabel('absolute error')
    plt.legend()
    plt.savefig('romberg_compare.png')
    print('plot saved as romberg_compare.png')


def rombergTrap(a, b, nrefine, f):
    ntest = 2**np.arange(0, nrefine+1)
    nn = len(ntest)
    Rom = np.zeros((nn, nn))

    for j in range(0, nn):
        Rom[j, 0] = CompTrap(a, b, ntest[j], f)

    # trapezoid error: h^2, h^4, ... -> divide by 4^j - 1
    for j in range(1, nn):
        for i in range(j, nn):
            Rom[i, j] = Rom[i, j-1] + (Rom[i, j-1] - Rom[i-1, j-1])/(4**j - 1)

    return Rom


def rombergSimp(a, b, nrefine, f):
    ntest = 2**np.arange(1, nrefine+2)
    nn = len(ntest)
    Rom = np.zeros((nn, nn))

    for j in range(0, nn):
        Rom[j, 0] = CompSimp(a, b, ntest[j], f)

    # Simpson error: h^4, h^6, ... -> divide by 4^(j+1) - 1
    for j in range(1, nn):
        for i in range(j, nn):
            Rom[i, j] = Rom[i, j-1] + (Rom[i, j-1] - Rom[i-1, j-1])/(4**(j+1) - 1)

    return Rom


def CompTrap(a, b, n, f):
    h = (b - a)/n
    xnode = a + np.arange(0, n+1)*h
    return h*(f(xnode[0])/2 + np.sum(f(xnode[1:n])) + f(xnode[n])/2)


def CompSimp(a, b, n, f):
    h = (b - a)/n
    xnode = a + np.arange(0, n+1)*h
    return h/3*(f(xnode[0]) + 4*np.sum(f(xnode[1:n:2])) + 2*np.sum(f(xnode[2:n:2])) + f(xnode[n]))


driver()