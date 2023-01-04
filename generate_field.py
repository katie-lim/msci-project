# %%
import pyshtools as pysh

import numpy as np
from scipy.special import spherical_jn

from generate_f_lmn import generate_f_lmn
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros


l_max = 100
k_max = 50
r_max = 5
n_max = 1000

# l_max = 3
# k_max = 10
# r_max = 5
# n_max = 1000

c_ln_values = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")
f_lmn = generate_f_lmn(l_max, n_max, r_max)


def a_lm(r_i, l, m, k_max, r_max):
    s = 0

    k_ln = 0
    n = 0

    while k_ln < k_max:

        k_ln = sphericalBesselZeros[l][n] / r_max

        s += ((r_max)**(-3/2)) * c_ln_values[l][n] * spherical_jn(l, k_ln * r_i) * f_lmn[l][m][n]

        n += 1

    return s


def calcCoeffs(r_i, l_max, k_max, r_max):
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

# Do the spherical harmonic transform with pyshtools

r_i = 3
lmax_calc = l_max
title = "r_i = %.2f, r_max = %.2f, k_max = %.2f, l_max = %d, lmax_calc = %d" % (r_i, r_max, k_max, l_max, lmax_calc)


# Get the coefficients a_lm
# in the format required by pyshtools
cilm = calcCoeffs(r_i, l_max, k_max, r_max)
coeffs = pysh.SHCoeffs.from_array(cilm)
# coeffs = pysh.SHCoeffs.from_array(cilm, normalization="ortho", csphase=-1)
# coeffs = pysh.SHCoeffs.from_array(cilm, normalization="ortho")


# Do the transform
grid = coeffs.expand()
# grid = coeffs.expand(lmax_calc=lmax_calc)


# Plot the field
fig = grid.plot(colorbar='right', title=title, show=False)


# Plot the field using the Mollweide projection
fig = grid.plotgmt(projection='mollweide', colorbar='right', title=title)
fig.show()


# %%

# Try doing the reverse transform to see if we get back the same a_lm

result = grid.expand()
# result = grid.expand(lmax_calc=lmax_calc, normalization="ortho", csphase=-1)


# %%

result


# %%

# result.coeffs

# # %%

# cilm

# # %%

# result.coeffs - cilm


# # %%

# np.max(result.coeffs - cilm)


# # %%

# cilm[:, :91, :91]


# # %%

# result.coeffs - cilm[:, :91, :91]


# %%


# %%

