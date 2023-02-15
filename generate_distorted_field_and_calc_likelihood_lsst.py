# %%
import numpy as np
from os import path

from utils import calc_n_max_l, plotField
from generate_field import generateTrueField, multiplyFieldBySelectionFunction
from spherical_bessel_transform import calc_f_lmn_0
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from compute_likelihood_lsst import calc_all_W, calc_all_S, computeLikelihood
from plot_likelihood_results import plotLikelihoodResults
from lsst_redshift_dist import calc_nbar, calc_phi_of_r0

from distance_redshift_relation import *


l_max = 15
k_min = 0
k_max = 300
r_max_true = 0.75
n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0
nbar = calc_nbar()


c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

# %%

# First, generate a true fluctuation field

omega_matter_true = 0.315
radii_true = np.linspace(0, r_max_true, 1001)


z_true, all_grids = generateTrueField(radii_true, omega_matter_true, r_max_true, l_max, k_max)

# %%

# Plot the field at some different radii

for i in range(1, len(radii_true), 100):
    print(i)
    plotField(all_grids[i], radii_true[i], r_max_true, k_max, l_max, l_max)


# %%

# Add the effect of the selection function
# TODO: Check this

phiOfR0 = calc_phi_of_r0(omega_matter_true)

radii_true, all_observed_grids = multiplyFieldBySelectionFunction(radii_true, all_grids, phiOfR0)

# Multiply the fractional fluctuation field by nbar, to give the number density field
all_observed_grids = [grid * nbar for grid in all_observed_grids]

# %%
# --------------- OBSERVED


omega_matter_0 = 0.315

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)
r_max_0 = radii_fiducial[-1]

# %%

f_lmn_0 = calc_f_lmn_0(radii_fiducial, all_observed_grids, l_max, k_max, n_max)

print(f_lmn_0)

# %%

saveFileName = "data/f_lmn_0_lsst_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true)

np.save(saveFileName, f_lmn_0)
print("Done! File saved to %s" % saveFileName)


# %%

# Calculate likelihood

omega_matters = np.linspace(omega_matter_0 - 0.01, omega_matter_0 + 0.01, 21)
logLikelihoods = []

# %%

for omega_matter in omega_matters:

    W_saveFileName = "data/W_lsst_omega_m-%.4f_omega_m_0-%.4f_l_max-%d_k_max-%.2f_r_max_0-%.4f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0)
    S_saveFileName = "data/S_lsst_omega_m-%.4f_omega_m_0-%.4f_l_max-%d_k_max-%.2f_r_max_0-%.4f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0)

    if not path.exists(W_saveFileName):
        print("Computing W's for Ωₘ = %.4f." % omega_matter)
        r0OfR = getInterpolatedR0ofR(omega_matter_0, omega_matter)
        W = calc_all_W(l_max, k_max, r_max_0, phiOfR0, r0OfR)
        np.save(W_saveFileName, W)

    if not path.exists(S_saveFileName):
        print("Computing S's for Ωₘ = %.4f." % omega_matter)
        S = calc_all_S(l_max, k_max, r_max_0, phiOfR0)
        np.save(S_saveFileName, S)



logLikelihoods = []

for omega_matter in omega_matters:

    W_saveFileName = "data/W_lsst_omega_m-%.4f_omega_m_0-%.4f_l_max-%d_k_max-%.2f_r_max_0-%.4f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0)
    S_saveFileName = "data/S_lsst_omega_m-%.4f_omega_m_0-%.4f_l_max-%d_k_max-%.2f_r_max_0-%.4f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0)

    W = np.load(W_saveFileName)
    S = np.load(S_saveFileName)

    print("Computing likelihood for Ωₘ = %.4f." % omega_matter)
    likelihood = computeLikelihood(f_lmn_0, k_min, k_max, r_max_0, W, S, nbar)
    logLikelihoods.append(likelihood)


# %%

# Calculate the redshift limit equivalent to the radial limit
# (assuming the fiducial cosmology)
z_max = getInterpolatedZofR(omega_matter_0)(r_max_0)

title = "LSST, $\Omega_m^{true}$=%.3f\n$\Omega_m^{fiducial}}$=%.3f\n$l_{max}$=%d, $k_{min}$=%.1f, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_min, k_max, r_max_0, z_max, n_max)

# Plot and analyse the log likelihood function

plotLikelihoodResults(omega_matters, logLikelihoods, omega_matter_true, title)

# %%
