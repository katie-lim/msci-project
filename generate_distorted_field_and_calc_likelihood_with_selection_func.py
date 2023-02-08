# %%
import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import curve_fit
from os import path

from utils import calc_n_max_l, computeIntegralSplit, gaussianPhi, plotField
from generate_field import generateTrueField, multiplyFieldBySelectionFunction
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from compute_likelihood_selection_func_shot_noise import calc_all_W_1st_terms, calc_all_W_2nd_terms_without_delta_omega_m, computeLikelihood

from distance_redshift_relation import *


l_max = 15
k_min = 100
k_max = 200
r_max_true = 0.75
n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0
R = 0.25 # Selection function scale length
nbar = 5


c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

# %%

# First, generate a true field

omega_matter_true = 0.5
radii_true = np.linspace(0, r_max_true, 1000)


z_true, all_grids = generateTrueField(radii_true, omega_matter_true, r_max_true, l_max, k_max)

# %%

# Plot the field at some different radii

for i in range(1, len(radii_true), 100):
    print(i)
    plotField(all_grids[i], radii_true[i], r_max_true, k_max, l_max, l_max)


# %%

# Add the effect of the selection function
def phi(r):
    return gaussianPhi(r, R)

radii_true, all_observed_grids = multiplyFieldBySelectionFunction(radii_true, all_grids, phi)


# %%
# --------------- OBSERVED


omega_matter_0 = 0.50

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)
r_max_0 = radii_fiducial[-1]


all_fiducial_coeffs = []

for i in range(len(radii_fiducial)):
    grid = all_observed_grids[i]

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

# l_test, m_test = 14, 10
l_test, m_test = 15, 15

plt.figure(dpi=200)
plt.plot(radii_fiducial, a_lm_real_interps[l_test][m_test](radii_fiducial), label="real")
plt.plot(radii_fiducial, a_lm_imag_interps[l_test][m_test](radii_fiducial), label="imag")
plt.xlabel("r_0")
plt.title("a_%d,%d(r_0)" % (l_test, m_test))
plt.legend()
plt.show()


# %%

f_lmn_0 = np.zeros((l_max + 1, l_max + 1, n_max + 1), dtype=complex)


for l in range(l_max + 1):
    n_max_l = calc_n_max_l(l, k_max, r_max_0) # Will using r_max_0 instead of r_max change the number of modes?

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

            real_integral = computeIntegralSplit(real_integrand, 50, r_max_0)
            imag_integral = computeIntegralSplit(imag_integrand, 50, r_max_0)

            total_integral = real_integral + (1j * imag_integral)

            f_lmn_0[l][m][n] = c_ln * total_integral


# %%

print(f_lmn_0)

# %%

saveFileName = "f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_with_phi" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R)

np.save(saveFileName, f_lmn_0)
print("Done! File saved to %s" % saveFileName)

# %%

# Or, load f_lmn_0 from a file
omega_matter_true = 0.5
omega_matter_0 = 0.5
l_max = 15
k_max = 300
r_max_true = 0.75
R = 0.25

saveFileName = "f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_with_phi.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R)

f_lmn_0 = np.load(saveFileName)

# f_lmn_0_loaded = np.load("f_lmn_0_true-0.5_fiducial-0.48_l_max-15_k_max-100_r_max_true-0.8.npy")
# f_lmn_0_loaded = np.load("f_lmn_0_true-0.500_fiducial-0.480_l_max-15_k_max-100.00_r_max_true-0.800.npy")


# %%

# Plot an example integrand (uncomment this code)

# l, m, n = 1, 1, 6

# k_ln = sphericalBesselZeros[l][n] / r_max_0
# c_ln = ((r_max)**(-3/2)) * c_ln_values_without_r_max[l][n]

# def real_integrand(r0):
#     return spherical_jn(l, k_ln * r0) * r0*r0 * a_lm_real_interps[l][m](r0)

# def imag_integrand(r0):
#     return spherical_jn(l, k_ln * r0) * r0*r0 * a_lm_imag_interps[l][m](r0)


# plt.plot(radii_fiducial, real_integrand(radii_fiducial))
# plt.show()

# print("Quad:", quad(real_integrand, 0, r_max_0))
# print("Integral split into 10 pieces:", computeIntegralSplit(real_integrand, 10, r_max_0))


# %%


# Calculate likelihood

omega_matters = np.linspace(omega_matter_0 - 0.015, omega_matter_0 + 0.015, 71)

# %%

dr_domega = getPartialRbyOmegaMatterInterp(omega_matter_0)

# Compute the W's
# If we've already computed them for these parameters before, then simply load them
# If not, compute the W's for these parameters and save them for future re-use

W_1st_terms_saveFileName = "W_1st_terms_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (l_max, k_max, r_max_0, R)
W_2nd_terms_without_delta_omega_m_saveFileName = "W_2nd_terms_without_delta_omega_m_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (l_max, k_max, r_max_0, R)


if path.exists(W_1st_terms_saveFileName):
    W_1st_terms = np.load(W_1st_terms_saveFileName)
else:
    W_1st_terms = calc_all_W_1st_terms(l_max, k_max, r_max_0, R)
    np.save(W_1st_terms_saveFileName, W_1st_terms)


if path.exists(W_2nd_terms_without_delta_omega_m_saveFileName):
    W_2nd_terms_without_delta_omega_m = np.load(W_2nd_terms_without_delta_omega_m_saveFileName)
else:
    W_2nd_terms_without_delta_omega_m = calc_all_W_2nd_terms_without_delta_omega_m (l_max, k_max, r_max_0, R, dr_domega)
    np.save(W_2nd_terms_without_delta_omega_m_saveFileName, W_2nd_terms_without_delta_omega_m)



# %%

likelihoods = [computeLikelihood(f_lmn_0, k_min, k_max, r_max_0, omega_m, omega_matter_0, W_1st_terms, W_2nd_terms_without_delta_omega_m, nbar) for omega_m in omega_matters]

# Convert from complex numbers to floats
likelihoods = np.real(likelihoods)

# %%

# Calculate the redshift limit equivalent to the radial limit
# (assuming the fiducial cosmology)
z_max = getInterpolatedZofR(omega_matter_0)(r_max_0)


# Plot the log likelihood function

plt.figure(dpi=200)
plt.plot(omega_matters, likelihoods)
# plt.plot(omega_matters, likelihoods, '.')
plt.xlabel("$\Omega_m$")
plt.ylabel("ln L")
plt.title("ln L($\Omega_m$)\n$\Omega_m^{true}$=%.2f\n$\Omega_m^{fiducial}}$=%.2f\n$l_{max}$=%d, $k_{min}$=%.1f, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $R$=%.3f, $\overline{n}$=%.1f, $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_min, k_max, r_max_0, z_max, R, nbar, n_max))
plt.show()
# %%


# Find the maximum
peak_index = np.argmax(likelihoods)
omega_m_peak = omega_matters[peak_index]
print("Peak is at Ωₘ = %.4f" % omega_m_peak)

# Find the index of the true Ωₘ
true_index = np.argmin(np.abs(omega_matters - omega_matter_true))

print("ln L(true Ωₘ) = %.3f" % np.real(likelihoods[true_index]))
print("ln L(peak Ωₘ) = %.3f" % np.real(likelihoods[peak_index]))
print("ln L(true Ωₘ) - ln L(peak Ωₘ) = %.3f" % np.real(likelihoods[true_index] - likelihoods[peak_index]))
print("L(true Ωₘ) / L(peak Ωₘ) = %.3e" % np.exp(np.real(likelihoods[true_index] - likelihoods[peak_index])))

# %%

# Plot the likelihood
lnL_peak = likelihoods[peak_index]
delta_lnL = likelihoods - lnL_peak


plt.figure(dpi=200)
plt.plot(omega_matters, np.exp(delta_lnL))
plt.xlabel("$\Omega_m$")
plt.ylabel("L/L$_{peak}$")
plt.title("L($\Omega_m$)/L$_{peak}$\n$\Omega_m^{true}$=%.2f\n$\Omega_m^{fiducial}}$=%.2f\n$l_{max}$=%d, $k_{min}$=%.1f, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $R$=%.3f, $\overline{n}$=%.1f, $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_min, k_max, r_max_0, z_max, R, nbar, n_max))
plt.show()

# %%

# Estimate the width, sigma

def quadratic(x, sigma):
    return -1/2 * ((x - omega_m_peak)/sigma)**2


params, cov = curve_fit(quadratic, omega_matters, delta_lnL, [1])
sigma = np.abs(params[0])

print("σ = %.5f" % sigma)

# %%

plt.figure(dpi=200)
plt.plot(omega_matters, delta_lnL, ".", label="$\Delta$ ln L")
plt.plot(omega_matters, quadratic(omega_matters, *params), label="Gaussian fit")

plt.xlabel("$\Omega_m$")
plt.ylabel("$\Delta$ ln L")
# plt.title("$\Delta$ ln L($\Omega_m$)\n$\Omega_m^{true}$=%.2f\n$\Omega_m^{fiducial}}$=%.2f\n$l_{max}$=%d, $k_{min}$=%.1f, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $R$=%.3f, $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_min, k_max, r_max_0, z_max, R, n_max))
plt.title("$\Delta$ ln L($\Omega_m$)\n$\Omega_m^{true}$=%.2f\n$\Omega_m^{fiducial}}$=%.2f\n$l_{max}$=%d, $k_{min}$=%.1f, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $R$=%.3f, $\overline{n}$=%.1f, $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_min, k_max, r_max_0, z_max, R, nbar, n_max))
plt.legend()
plt.show()


print("Result: Ωₘ = %.5f +/- %.5f" % (omega_m_peak, sigma))
# %%
