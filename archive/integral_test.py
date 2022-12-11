# %%
import scipy as sp
from scipy.special import spherical_jn


def computeIntegral(integrand, zeros):
    answer = 0

    for i in range(len(zeros)-1):
        answer += sp.integrate.quad(integrand, zeros[i], zeros[i+1])[0]

    return answer


# %%

# import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


# Test the integral code on the orthogonality relation



def test():
    l  = 100
    n1 = 50
    Integral = np.zeros(2*n1)

    k1 = sphericalBesselZeros[l][n1]

    zeros = [i/10 for i in range(11)]


    for n2 in range(1,2*n1):
        k2 = sphericalBesselZeros[l][n2]

        f = lambda z: z*z*spherical_jn(l, k1*z)*spherical_jn(l, k2*z)

        Answer = computeIntegral(f, zeros)

        Integral[n2] = Answer
        print(n2,Integral[n2])

    plt.plot(Integral)
    plt.xlabel("n2")
    plt.ylabel("integral")
    plt.title("n1 = %d" % n1)


test()

# Gives a delta function. Equals 1/c_ln^2 when n1 = n2

# %%
