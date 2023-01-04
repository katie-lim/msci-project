# %%
import pyshtools as pysh

import numpy as np
from scipy.special import spherical_jn

from generate_f_lmn import generate_f_lmn
from precompute_c_ln import get_c_ln_values_with_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

# l_max = 10
# n_max = 10000
# k_max = 30
# r_max = 100
l_max = 10
n_max = 10000
k_max = 5
r_max = 10

c_ln_values = get_c_ln_values_with_r_max("c_ln.csv", r_max)
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")
f_lmn = generate_f_lmn(l_max, n_max)


def a_lm(r_i, l, m, k_max, r_max):
    s = 0

    k_ln = 0
    n = 0

    while k_ln < k_max:

        k_ln = sphericalBesselZeros[l][n] / r_max

        s += c_ln_values[l][n] * spherical_jn(l, k_ln * r_i) * f_lmn[l][m][n]

        n += 1

    return s


def calcAllA_lm(r_i, l_max, k_max, r_max):
    cilm = np.zeros((2, l_max + 1, l_max + 1))

    for l in range(l_max + 1):

        # We don't need to supply the -ve m
        # Since we are working with real fields
        for m in range(l + 1):
            coeff = a_lm(r_i, l, m, k_max, r_max)

            cilm[0][l][m] = np.real(coeff)
            cilm[1][l][m] = np.imag(coeff)


    return cilm


# %%

cilm = calcAllA_lm(1, l_max, k_max, r_max)



# cilm

result = pysh.expand.MakeGridDH(cilm)



import matplotlib.pyplot as plt

# result = plt.imread("cat.jpg")
x = np.linspace(-np.pi,np.pi,result.shape[1])
y = np.linspace(-np.pi/2,np.pi/2,result.shape[0])
X,Y = np.meshgrid(x,y)

plt.subplot(111, projection="aitoff")
plt.pcolormesh(X,Y[::-1],result)
plt.show()

# %%
result
# %%

result.shape


# %%
