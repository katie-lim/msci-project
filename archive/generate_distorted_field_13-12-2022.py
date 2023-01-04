# %%
import pyshtools as pysh

import numpy as np
from scipy.special import spherical_jn, sph_harm

from utils import calc_n_max_l
from generate_f_lmn import generate_f_lmn
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

from distance_redshift_relation import *


l_max = 15
k_max = 100
r_max_true = 0.8
n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0


c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

# %%

omega_m_true = 0.5
f_lmn_true = generate_f_lmn(l_max, r_max_true, k_max)

# %%

def f_of_r(r_true, theta, phi, f_lmn_true, l_max, k_max, r_max):
    total = 0

    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max)

        for m in range(-l, l + 1):

            for n in range(n_max_l + 1):
                if m < 0:
                    f_lmn_value = np.conj(f_lmn_true[l][-m][n]) * (-1)**m
                else:
                    f_lmn_value = f_lmn_true[l][m][n]

                k_ln = sphericalBesselZeros[l][n] / r_max
                c_ln = ((r_max)**(-3/2)) * c_ln_values_without_r_max[l][n]

                total += f_lmn_value * c_ln * spherical_jn(l, k_ln * r_true) * sph_harm(m, l, theta, phi)

    return total

# %%

f_of_r(0.5, 1, 1, f_lmn_true, l_max, k_max, r_max)

# %%

f_of_r(0.5, 1, 2, f_lmn_true, l_max, k_max, r_max)

# %%

def f_of_z(z, rOfZ, theta, phi, f_lmn_true, l_max, k_max, r_max):
    total = 0

    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max)

        for m in range(-l, l + 1):

            for n in range(n_max_l + 1):
                if m < 0:
                    f_lmn_value = np.conj(f_lmn_true[l][-m][n]) * (-1)**m
                else:
                    f_lmn_value = f_lmn_true[l][m][n]

                k_ln = sphericalBesselZeros[l][n] / r_max
                c_ln = ((r_max)**(-3/2)) * c_ln_values_without_r_max[l][n]

                total += f_lmn_value * c_ln * spherical_jn(l, k_ln * rOfZ(z)) * sph_harm(m, l, theta, phi)

    return total


# %%

rOfZ = getInterpolatedRofZ(0.3)

f_of_z(0.2, rOfZ, 1, 1, f_lmn_true, l_max, k_max, r_max)


# %%

f_of_z(0.2, rOfZ, 0.3, 1, f_lmn_true, l_max, k_max, r_max)

# %%

f_of_z(0.2, rOfZ, 0.9, 4, f_lmn_true, l_max, k_max, r_max)

# %%

thetas = np.linspace(0, np.pi, 50)
phis = np.linspace(0, 2*np.pi, 100, endpoint=False)

radii_true = np.linspace(0, r_max_true, 1000)
true_z_of_r = getInterpolatedZofR(omega_m_true)
z_true = true_z_of_r(radii_true)


# %%


def a_lm(r_i, l, m, k_max, r_max, f_lmn):
    s = 0

    n = 0
    k_ln = sphericalBesselZeros[l][0] / r_max

    while k_ln < k_max:

        s += ((r_max)**(-3/2)) * c_ln_values_without_r_max[l][n] * spherical_jn(l, k_ln * r_i) * f_lmn[l][m][n]

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



# Calculate the spherical harmonic coefficients
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


# -----


omega_matter_0 = 0.48

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)


all_fiducial_coeffs = []

for i in range(len(radii_fiducial)):
    grid = all_grids[i]

    fiducial_coeffs = grid.expand()
    all_fiducial_coeffs.append(fiducial_coeffs)


# %%

all_fiducial_coeffs

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
    n_max_l = calc_n_max_l(l, k_max, r_max_0) # r_max_0?

    print("l = %d" % l)

    for m in range(l + 1):
        for n in range(n_max_l + 1):
            k_ln = sphericalBesselZeros[l][n] / r_max_0
            c_ln = ((r_max_0)**(-3/2)) * c_ln_values_without_r_max[l][n]

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

f_lmn_0

# %%

np.save("f_lmn_0_values_true-0.5_fiducial-0.48.npy", f_lmn_0)


# %%

l, m, n = 1, 1, 6

k_ln = sphericalBesselZeros[l][n] / r_max_0
c_ln = ((r_max_0)**(-3/2)) * c_ln_values_without_r_max[l][n]

def real_integrand(r0):
    return spherical_jn(l, k_ln * r0) * r0*r0 * a_lm_real_interps[l][m](r0)

def imag_integrand(r0):
    return spherical_jn(l, k_ln * r0) * r0*r0 * a_lm_imag_interps[l][m](r0)



plt.plot(radii_fiducial, real_integrand(radii_fiducial))
plt.show()

print("Quad:", quad(real_integrand, 0, r_max_0))
print("Split integral:", computeIntegralSplit(real_integrand, 10, r_max_0))


# %%


# def generateFieldAtRadius(r_i, l_max, k_max, r_max, lmax_calc=None):
#     if lmax_calc == None:
#         # Default to l_max
#         lmax_calc = l_max


#     # Get the coefficients a_lm
#     # in the format required by pyshtools
#     cilm = calcCoeffs(r_i, l_max, k_max, r_max)
#     coeffs = pysh.SHCoeffs.from_array(cilm)


#     # Do the transform
#     grid = coeffs.expand(lmax_calc=lmax_calc)

#     return grid


# %%




# Evaluate true field on a grid

field_grid = np.zeros((radii_true.size, thetas.size, phis.size))

for i in range(len(radii_true)):
    r_true = radii_true[i]
    for j in range(len(thetas)):
        theta = thetas[j]
        for k in range(len(phis)):
            phi = phis[k]

            field_grid[i][j][k] = f_of_r(r_true, theta, phi, f_lmn_true, l_max, k_max, r_max_true)



# %%


field_grid

# %%

f_of_r(0.5, 1, 1, f_lmn_true, l_max, k_max, r_max_true)

# %%

# %%

angular_integrals = []

for i in range(len(radii_fiducial)):
    field_grid_at_radius = field_grid[i]

    shgrid = pysh.SHGrid.from_array(field_grid_at_radius)
    
    fig = shgrid.plotgmt(projection='mollweide', colorbar='right')
    fig.show()

    angular_integrals.append(shgrid.expand(lmax_calc=l_max))

# %%

angular_integrals

# %%

angular_integrals[1]

# %%

angular_integrals[1].plot_spectrum2d()


# %%

angular_integrals[1].coeffs[0]

# %%

angular_integrals[1].coeffs[1]

# %%

# %%


# %%
