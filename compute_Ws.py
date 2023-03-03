# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn
from scipy.integrate import quad, simpson
from utils import calc_n_max_l, computeIntegralSimpson, computeIntegralSplit

from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


def getZerosOfJ_lUpToBoundary(l, upperLimit):
    n = 0
    root = sphericalBesselZeros[l][0]

    if root < upperLimit:
        while root < upperLimit:
            n += 1
            root = sphericalBesselZeros[l][n]

        n_max = n - 1

        return sphericalBesselZeros[l][:n_max + 1]
        
    else:
        return []


def calc_W_SplitIntegralByZeros(n, n_prime, l, r_max_0, r0OfR, rOfR0, phiOfR0, simpson=False, simpsonNpts=None, plot=False):
    k_ln = sphericalBesselZeros[l][n] / r_max_0
    k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max_0


    def W_integrand(r):
        r0 = r0OfR(r)

        return phiOfR0(r0) * spherical_jn(l, k_ln_prime*r) * spherical_jn(l, k_ln*r0) * r*r

    r_boundary = k_ln_prime * rOfR0(r_max_0)
    r0_boundary = k_ln * r_max_0

    r_zeros = getZerosOfJ_lUpToBoundary(l, r_boundary) / k_ln_prime
    r0_zeros = getZerosOfJ_lUpToBoundary(l, r0_boundary) / k_ln

    # Convert r0 values to r values
    r0_zeros = rOfR0(r0_zeros)

    # Combine and sort the zeros
    zeros = np.sort(np.append(r_zeros, r0_zeros))

    # Remove any duplicate zeros (which occur in the case r = r0)
    zeros = np.unique(zeros)


    zeros = np.append(zeros, [r_max_0])
    zeros = np.insert(zeros, 0, 0)


    if plot:
        x = np.linspace(0, r_max_0, 2000)
        y = W_integrand(x)

        plt.figure(dpi=200)
        plt.plot(x, y)
        plt.vlines(zeros, np.min(y), np.max(y), "r", "dotted")
        plt.show()


    integral = 0

    if simpson:
        for i in range(0, np.size(zeros) - 1):
            integral += computeIntegralSimpson(W_integrand, zeros[i], zeros[i+1], simpsonNpts)
    else:
        for i in range(0, np.size(zeros) - 1):
            integralChunk, error = quad(W_integrand, zeros[i], zeros[i+1])
            integral += integralChunk


    return np.power(r_max_0, -3) * c_ln_values_without_r_max[l][n] * c_ln_values_without_r_max[l][n_prime] * integral



def calc_W(n, n_prime, l, r_max_0, r0OfR, phiOfR0, Nsplit=10):
    k_ln = sphericalBesselZeros[l][n] / r_max_0
    k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max_0


    def W_integrand(r):
        r0 = r0OfR(r)

        return phiOfR0(r0) * spherical_jn(l, k_ln_prime*r) * spherical_jn(l, k_ln*r0) * r*r


    integral = computeIntegralSplit(W_integrand, Nsplit, r_max_0)

    return np.power(r_max_0, -3) * c_ln_values_without_r_max[l][n] * c_ln_values_without_r_max[l][n_prime] * integral




def calc_all_W(l_max, k_max, r_max_0, r0OfR, rOfR0, phiOfR0):
    # The maximum number of modes is when l=0
    n_max_0 = calc_n_max_l(0, k_max, r_max_0)

    W_lnn_prime = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1))


    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max_0)

        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):
                W_lnn_prime[l][n1][n2] = calc_W_SplitIntegralByZeros(n1, n2, l, r_max_0, r0OfR, rOfR0, phiOfR0)

    return W_lnn_prime

