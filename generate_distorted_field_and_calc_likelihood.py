# %%
import pyshtools as pysh

import numpy as np
from scipy.special import spherical_jn

from generate_f_lmn import generate_f_lmn
from precompute_c_ln import load_c_ln_values
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from compute_likelihood import calculate_n_max_l, calc_all_Ws_without_delta_omega_m, computeLikelihood

from distance_redshift_relation import *



l_max = 7
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

# Generate field at many radii, add distortion

# Set a "true" cosmology
omega_matter_true = 0.5

# Generate a set of radii (shells)
# r_vals_true = np.linspace(0.01, 2.5, 50)
r_vals_true = np.linspace(0.001, 2.5, 50)
# r_vals_true = np.linspace(0.001, 2.5, 50)

# Get the true redshift-distance relation
z_interp = getInterpolatedZofR(omega_matter_true)

# # Convert the true radii to true redshifts
# z_vals = z_interp(r_vals_true)
z_vals = np.linspace(0.00001, 1000, 5000)


# Generate an (angular) field at each true radius
field = []
lmax_calc = l_max

for r_val in r_vals_true:
    field_at_radius = generateFieldAtRadius(r_val)
    field.append(field_at_radius)


# %%

# Calculate the a_lm for the field

coeffs = []

for i in range(len(r_vals_true)):
    r_val = r_vals_true[i]
    field_at_radius = field[i]

    coeffs_at_radius = field_at_radius.expand()
    coeffs.append(coeffs_at_radius)



a_lm_interp = [[] for _ in range(l_max + 1)]

for l in range(l_max + 1):
    for m in range(l + 1):
        a_lm_values = []

        for i in range(len(r_vals_true)):
            coeffs_at_radius = coeffs[i].coeffs

            a_lm_real = coeffs_at_radius[0][l][m]
            a_lm_imag = coeffs_at_radius[1][l][m]

            a_lm_values.append(a_lm_real + (a_lm_imag * 1j))

        a_lm_interpolated = interp1d(r_vals_true, a_lm_values)
        # a_lm[l][m] = a_lm_interp
        a_lm_interp[l].append(a_lm_interpolated)


# Try plotting an interpolated a_lm
# l_to_plot, m_to_plot = 1, 1

# plt.plot(r_vals, np.real(a_lm_interp[l_to_plot][m_to_plot](r_vals)))
# plt.plot(r_vals, np.imag(a_lm_interp[l_to_plot][m_to_plot](r_vals)))
# plt.show()



# %%

# Assume a fiducial cosmology with a different omega_matter
omega_matter_0 = 0.5

# Get the distance-redshift relation in the fiducial cosmology
r_0_interp = getInterpolatedRofZ(omega_matter_0)

# Convert the true redshifts into measured radii
r_0_vals = r_0_interp(z_vals)

# %%


# Now we'll vary omega_matter.
# omega_matter = 0.51
# print("Calculating f_lmn^0 for Ωₘ = %.3f." % omega_matter)

# # Get the distance-redshift relation in the assumed true cosmology
# r_interp = getInterpolatedRofZ(omega_matter)
# # Convert the measured redshifts into assumed true radii
# r_vals = r_interp(z_vals)

# r_0_of_r_interp = interp1d(r_vals, r_0_vals)


# print("r_max =", r_max)
# print("maximum of interp:", r_vals[-1])
# print("minimum of interp:", r_vals[0])
# print("min:", r_0_of_r_interp(0.01))
# r_max_0 = r_0_of_r_interp(r_max)



# %%

def calc_f_lmn_0(omega_matter):

    # Now we'll vary omega_matter.
    print("Calculating f_lmn^0 for Ωₘ = %.3f." % omega_matter)

    # Get the distance-redshift relation in the assumed true cosmology
    r_interp = getInterpolatedRofZ(omega_matter)
    # Convert the measured redshifts into assumed true radii
    r_vals = r_interp(z_vals)

    r_0_of_r_interp = interp1d(r_vals, r_0_vals)


    # Plot the measured radii as a function of true radii
    # (r_0 as a function of r)

    # plt.plot(r_vals, r_0_vals)
    # plt.xlabel("r")
    # plt.ylabel("$r_0$")
    # plt.title("True: $\Omega_m$=%.2f, Fiducial: $\Omega_m^0$=%.2f" % (omega_matter, omega_matter_0))
    # plt.show()


    # Get the Jacobian
    jacobian = calculateJacobian(r_vals, r_0_vals)


    # Get f_lmn^0 coefficients in the fiducial cosmology
    f_lmn_0 = np.zeros((l_max + 1, l_max + 1, n_max + 1), dtype=complex)

    print("r_max =", r_max)
    print("maximum of interp:", r_vals[-1])
    print("minimum of interp:", r_vals[0])
    print("min:", r_0_of_r_interp(0.01))
    r_max_0 = r_0_of_r_interp(r_max)


    for l in range(l_max + 1):
        print("l = %d" % l)
        for m in range(l + 1):

            n_max_l = calculate_n_max_l(l, k_max, r_max)

            for n in range(n_max_l + 1):

                k_ln = sphericalBesselZeros[l][n] / r_max_0
                c_ln = c_ln_values[l][n] * r_max_0**(-3/2)

                def integrand(r_val):
                    print("a_lm:")
                    print(a_lm_interp[l][m](r_val))
                    return a_lm_interp[l][m](r_val)
                    # return a_lm_interp[l][m](r_val) * spherical_jn(l, k_ln*r_val) * jacobian(r_val) * r_val**2


                # change lower limit to 0 later?
                lower_bound = r_0_of_r_interp(0.01)
                integral, error = quad(integrand, lower_bound, r_max_0)

                f_lmn_0[l][m][n] = c_ln * integral

    return f_lmn_0


# %%

# Compute the likelihood function

# omega_matters = np.linspace(0.48, 0.52, 30)
omega_matters = np.linspace(0.51, 0.52, 5)

print(omega_matters)

# %%

# Perform the inverse transform
f_lmn_0_vals = []

for omega_m in omega_matters:
    f_lmn_0_vals.append(calc_f_lmn_0(omega_m))


# %%

dr_domega = getPartialRbyOmegaMatterInterp(omega_matter_0)
Ws_without_delta_omega_m = calc_all_Ws_without_delta_omega_m(l_max, k_max, r_max, dr_domega)

# %%

# likelihoods = [computeLikelihood(f_lmn_0_vals[i], k_max, r_max, omega_matters[i], omega_matter_0, Ws_without_delta_omega_m) for i in range(len(omega_matters))]
likelihoods = [computeLikelihood(f_lmn_0_vals[i], k_max, r_max, omega_matters[i], omega_matter_0, Ws_without_delta_omega_m) for i in range(len(f_lmn_0_vals))]


# %%

omega_matters = np.linspace(0.48, 0.52, 30)
plt.plot(omega_matters[:len(likelihoods)], likelihoods)
plt.xlabel("$\Omega_m$")
plt.ylabel("ln L")
plt.show()


# %%
