# %%
import numpy as np
from os import path

from utils import calc_n_max_l, gaussianPhi, plotField
from generate_field import generateTrueField, multiplyFieldBySelectionFunction
from spherical_bessel_transform import calc_f_lmn_0
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from compute_likelihood_selection_func_shot_noise import calc_all_W_1st_terms, calc_all_W_2nd_terms_without_delta_omega_m, computeLikelihood
from plot_likelihood_results import plotLikelihoodResults
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

# %%

f_lmn_0 = calc_f_lmn_0(radii_fiducial, all_grids, l_max, k_max, n_max)

print(f_lmn_0)

# %%

saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_with_phi" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R)

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

saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_with_phi.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R)

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

logLikelihoods = [computeLikelihood(f_lmn_0, k_min, k_max, r_max_0, omega_m, omega_matter_0, W_1st_terms, W_2nd_terms_without_delta_omega_m, nbar) for omega_m in omega_matters]

# %%

# Calculate the redshift limit equivalent to the radial limit
# (assuming the fiducial cosmology)
z_max = getInterpolatedZofR(omega_matter_0)(r_max_0)

title = "\n$\Omega_m^{true}$=%.2f\n$\Omega_m^{fiducial}}$=%.2f\n$l_{max}$=%d, $k_{min}$=%.1f, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $R$=%.3f, $\overline{n}$=%.1f, $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_min, k_max, r_max_0, z_max, R, nbar, n_max)


# Plot and analyse the log likelihood function

plotLikelihoodResults(omega_matters, logLikelihoods, omega_matter_true, title)

# %%