# %%
import pyshtools as pysh

import numpy as np
from scipy.special import spherical_jn

from generate_f_lmn import generate_f_lmn
from precompute_c_ln import load_c_ln_values
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from compute_likelihood import calculate_n_max_l

from distance_redshift_relation import *


# l_max = 100
# k_max = 50
# r_max = 2.5
# n_max = 1000

# l_max = 3
# k_max = 10
# r_max = 5
# n_max = 1000

l_max = 15
k_max = 10
r_max = 2.5
n_max = calculate_n_max_l(0, k_max, r_max) # There are the most modes when l=0


c_ln_values = load_c_ln_values("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")
f_lmn = generate_f_lmn(l_max, n_max, r_max)


def a_lm(r_i, l, m, k_max, r_max):
    s = 0

    n = 0
    k_ln = sphericalBesselZeros[l][0] / r_max

    while k_ln < k_max:

        s += ((r_max)**(-3/2)) * c_ln_values[l][n] * spherical_jn(l, k_ln * r_i) * f_lmn[l][m][n]

        n += 1
        k_ln = sphericalBesselZeros[l][n] / r_max

    # print("l = %d, max n = %d" % (l, n - 1))

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


def generateFieldAtRadius(r_i, lmax_calc=None):
    if lmax_calc == None:
        # Default to l_max
        lmax_calc = l_max


    # Get the coefficients a_lm
    # in the format required by pyshtools
    cilm = calcCoeffs(r_i, l_max, k_max, r_max)
    coeffs = pysh.SHCoeffs.from_array(cilm)


    # Do the transform
    grid = coeffs.expand(lmax_calc=lmax_calc)

    return grid



def plotField(grid, r_i, lmax_calc):

    title = "r_i = %.2f, r_max = %.2f, k_max = %.2f, l_max = %d, lmax_calc = %d" % (r_i, r_max, k_max, l_max, lmax_calc)

    # Plot the field using the Mollweide projection

    fig = grid.plotgmt(projection='mollweide', colorbar='right', title=title)
    fig.show()





# %%

# Do the spherical harmonic transform with pyshtools

r_i = 3
lmax_calc = 15


grid = generateFieldAtRadius(r_i, lmax_calc)
plotField(grid, r_i, lmax_calc)


# %%

# Generate field at many radii, add distortion

# Set a "true" cosmology
omega_matter = 0.5

# Assume a fiducial cosmology with a different omega_matter
omega_matter_0 = 0.5


# Generate a set of radii (shells)
r_vals = np.linspace(0.01, 2.5, 50)


# Get the true redshift-distance relation
z_interp = getInterpolatedZofR(omega_matter)

# Convert the true radii to true redshifts
z_vals = z_interp(r_vals)


# Get the distance-redshift relation in the fiducial cosmology
r_interp = getInterpolatedRofZ(omega_matter_0)


# Convert the true redshifts into measured radii
r_0_vals = r_interp(z_vals)


r_0_of_r_interp = interp1d(r_vals, r_0_vals)


# Plot the measured radii as a function of true radii
# (r_0 as a function of r)

plt.plot(r_vals, r_0_vals)
plt.xlabel("r")
plt.ylabel("$r_0$")
plt.title("True: $\Omega_m$=%.2f, Fiducial: $\Omega_m^0$=%.2f" % (omega_matter, omega_matter_0))
plt.show()

# %%

# Generate a field at each radius
field = []
lmax_calc = l_max

for r_val in r_vals:
    print(r_val)
    field_at_radius = generateFieldAtRadius(r_val)

    field.append(field_at_radius)
    # plotField(field_at_radius, r_val, lmax_calc)



# %%


# Perform the inverse transform
coeffs = []

for i in range(len(r_vals)):
    r_val = r_vals[i]
    field_at_radius = field[i]

    coeffs_at_radius = field_at_radius.expand()
    coeffs.append(coeffs_at_radius)



# %%

coeffs

# %%

a_lm_interp = [[] for _ in range(l_max + 1)]

for l in range(l_max + 1):
    for m in range(l + 1):
        a_lm_values = []

        for i in range(len(r_vals)):
            coeffs_at_radius = coeffs[i].coeffs

            a_lm_real = coeffs_at_radius[0][l][m]
            a_lm_imag = coeffs_at_radius[1][l][m]

            a_lm_values.append(a_lm_real + (a_lm_imag * 1j))

        a_lm_interpolated = interp1d(r_vals, a_lm_values)
        # a_lm[l][m] = a_lm_interp
        a_lm_interp[l].append(a_lm_interpolated)


# %%

# Try plotting an interpolated a_lm
l_to_plot, m_to_plot = 1, 1

plt.plot(r_vals, np.real(a_lm_interp[l_to_plot][m_to_plot](r_vals)))
plt.plot(r_vals, np.imag(a_lm_interp[l_to_plot][m_to_plot](r_vals)))
plt.show()


# %%

# Get the Jacobian
jacobian = calculateJacobian(r_vals, r_0_vals)

# %%

# Get f_lmn coefficients
f_lmn_0 = np.zeros((l_max + 1, l_max + 1, n_max + 1), dtype=complex)

r_max_0 = r_0_of_r_interp(r_max)


for l in range(l_max + 1):
    print("l = %d" % l)
    for m in range(l + 1):
        n_max_l = calculate_n_max_l(l, k_max, r_max)

        for n in range(n_max_l + 1):

            k_ln = sphericalBesselZeros[l][n] / r_max_0
            c_ln = c_ln_values[l][n] * r_max_0**(-3/2)

            def integrand(r_val):
                return a_lm_interp[l][m](r_val) * spherical_jn(l, k_ln*r_val) * jacobian(r_val) * r_val**2


            # change lower limit to 0 later?
            integral, error = quad(integrand, r_0_of_r_interp(0.01), r_max_0)

            f_lmn_0[l][m][n] = c_ln * integral



# Save to a file, so we can read it back later
# without having to redo all the calculations
# np.save("f_lmn_0_values_15-11-2022.npy", f_lmn_0)

print("Done.")


# %%

print(f_lmn_0)

# %%

f_lmn_0_loaded = np.load("f_lmn_0_values_15-11-2022.npy")

print(f_lmn_0_loaded)


# %%

f_lmn_0_loaded.shape

# %%
