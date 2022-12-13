# %%
import pyshtools as pysh

import numpy as np
from scipy.special import spherical_jn, sph_harm

from generate_f_lmn import generate_f_lmn
from precompute_c_ln import load_c_ln_values
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from compute_likelihood import calculate_n_max_l

from distance_redshift_relation import *


l_max = 15
k_max = 100
r_max_true = 0.8
n_max = calculate_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0


c_ln_values = load_c_ln_values("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

# %%

omega_m_true = 0.5
f_lmn_true = generate_f_lmn(l_max, n_max, r_max_true)

# %%

radii_true = np.linspace(0, r_max_true, 1000)
true_z_of_r = getInterpolatedZofR(omega_m_true)
z_true = true_z_of_r(radii_true)


# %%


def a_lm(r_i, l, m, k_max, r_max, f_lmn):
    s = 0

    n = 0
    k_ln = sphericalBesselZeros[l][0] / r_max

    while k_ln < k_max:

        s += ((r_max)**(-3/2)) * c_ln_values[l][n] * spherical_jn(l, k_ln * r_i) * f_lmn[l][m][n]

        n += 1
        k_ln = sphericalBesselZeros[l][n] / r_max


    return s


def calcCoeffs(r_i, l_max, k_max, r_max, f_lmn):
    cilm = np.zeros((2, l_max + 1, l_max + 1))

    for l in range(l_max + 1):

        # We don't need to supply the -ve m
        # Since we are working with real fields
        for m in range(l + 1):
            coeff = a_lm(r_i, l, m, k_max, r_max, f_lmn)

            cilm[0][l][m] = np.real(coeff)
            cilm[1][l][m] = np.imag(coeff)


    return cilm



# Calculate the spherical harmonic coefficients for each shell
all_coeffs = []

for i in range(len(radii_true)):
    r_true = radii_true[i]

    cilm = calcCoeffs(r_true, l_max, k_max, r_max_true, f_lmn_true)
    coeffs = pysh.SHCoeffs.from_array(cilm)

    all_coeffs.append(coeffs)


# Expand the coefficients & evaluate the field on a grid
all_grids = []

for i in range(len(radii_true)):
    grid = all_coeffs[i].expand()

    all_grids.append(grid)


# ----- OBSERVED


omega_matter_0 = 0.48

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)


all_fiducial_coeffs = []

for i in range(len(radii_fiducial)):
    grid = all_grids[i]

    fiducial_coeffs = grid.expand()
    all_fiducial_coeffs.append(fiducial_coeffs)

# %%

a_lm_real_interps = []
a_lm_imag_interps = []

for l in range(l_max + 1):
    a_l_real_interps = []
    a_l_imag_interps = []

    for m in range(l + 1):
        real_parts = []
        imag_parts = []

        for i in range(len(radii_fiducial)):
            coeffs = all_fiducial_coeffs[i].coeffs

            real_parts.append(coeffs[0][l][m])
            imag_parts.append(coeffs[1][l][m])

        a_lm_interp_real = interp1d(radii_fiducial, real_parts)
        a_lm_interp_imag = interp1d(radii_fiducial, imag_parts)

        a_l_real_interps.append(a_lm_interp_real)
        a_l_imag_interps.append(a_lm_interp_imag)

    a_lm_real_interps.append(a_l_real_interps)
    a_lm_imag_interps.append(a_l_imag_interps)


# %%


# Plot some example a_lm(r)'s

# l_test, m_test = 0, 0
# l_test, m_test = 1, 0
# l_test, m_test = 1, 1
# l_test, m_test = 2, 0
# l_test, m_test = 2, 1
# l_test, m_test = 2, 2
# l_test, m_test = 3, 0
# l_test, m_test = 3, 1
# l_test, m_test = 3, 2
# l_test, m_test = 3, 3

l_test, m_test = 14, 10

plt.plot(radii_fiducial, a_lm_real_interps[l_test][m_test](radii_fiducial), label="real")
plt.plot(radii_fiducial, a_lm_imag_interps[l_test][m_test](radii_fiducial), label="imag")
plt.xlabel("r_0")
plt.title("a_%d,%d(r_0)" % (l_test, m_test))
plt.legend()
plt.show()


# %%


def computeIntegralSplit(integrand, N, upperLimit):
    answer = 0
    step = upperLimit / N

    for i in range(N):
        answer += quad(integrand, i*step, (i+1)*step)[0]

    return answer


# %%

r_max_0 = radii_fiducial[-1]
f_lmn_0 = np.zeros((l_max + 1, l_max + 1, n_max + 1), dtype=complex)


for l in range(l_max + 1):
    n_max_l = calculate_n_max_l(l, k_max, r_max_0) # Will using r_max_0 instead of r_max change the number of modes?

    print("l = %d" % l)

    for m in range(l + 1):
        for n in range(n_max_l + 1):
            k_ln = sphericalBesselZeros[l][n] / r_max_0
            c_ln = c_ln_values[l][n]

            def real_integrand(r0):
                return spherical_jn(l, k_ln * r0) * r0*r0 * a_lm_real_interps[l][m](r0)

            def imag_integrand(r0):
                return spherical_jn(l, k_ln * r0) * r0*r0 * a_lm_imag_interps[l][m](r0)


            # real_integral, error = quad(real_integrand, 0, r_max_0)
            # imag_integral, error = quad(imag_integrand, 0, r_max_0)

            real_integral = computeIntegralSplit(real_integrand, 10, r_max_0)
            imag_integral = computeIntegralSplit(imag_integrand, 10, r_max_0)

            total_integral = real_integral + (1j * imag_integral)

            f_lmn_0[l][m][n] = c_ln * total_integral


# %%

print(f_lmn_0)

# %%

np.save("f_lmn_0_values_true-0.5_fiducial-0.48.npy", f_lmn_0)


# %%

# Plot an example integrand (uncomment this code)

# l, m, n = 1, 1, 6

# k_ln = sphericalBesselZeros[l][n] / r_max_0
# c_ln = c_ln_values[l][n]

# def real_integrand(r0):
#     return spherical_jn(l, k_ln * r0) * r0*r0 * a_lm_real_interps[l][m](r0)

# def imag_integrand(r0):
#     return spherical_jn(l, k_ln * r0) * r0*r0 * a_lm_imag_interps[l][m](r0)


# plt.plot(radii_fiducial, real_integrand(radii_fiducial))
# plt.show()

# print("Quad:", quad(real_integrand, 0, r_max_0))
# print("Integral split into 10 pieces:", computeIntegralSplit(real_integrand, 10, r_max_0))


# %%