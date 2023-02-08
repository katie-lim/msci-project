# %%
import numpy as np
from os import path

from utils import calc_n_max_l
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from compute_likelihood_selection_func_shot_noise_no_tayl_exp import calc_all_W, computeLikelihood
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

true_z_of_r = getInterpolatedZofR(omega_matter_true)
z_true = true_z_of_r(radii_true)

# %%

# --------------- OBSERVED

omega_matter_0 = 0.5

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)
r_max_0 = radii_fiducial[-1]

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


# %%

# Calculate likelihood

omega_matters = np.linspace(omega_matter_0 - 0.01, omega_matter_0 + 0.01, 21)
logLikelihoods = []

# %%

for omega_matter in omega_matters:

    W_saveFileName = "W_no_tayl_exp_omega_m-%.3f_omega_m_0-%.3f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)

    if path.exists(W_saveFileName):
        W = np.load(W_saveFileName)
    else:
        r0OfR = getInterpolatedR0ofR(omega_matter_0, omega_matter)
        W = calc_all_W(l_max, k_max, r_max_0, R, r0OfR)
        np.save(W_saveFileName, W)

    likelihood = computeLikelihood(f_lmn_0, k_min, k_max, r_max_0, omega_m, omega_matter_0, W, nbar)
    logLikelihoods.append(likelihood)


# %%

# Calculate the redshift limit equivalent to the radial limit
# (assuming the fiducial cosmology)
z_max = getInterpolatedZofR(omega_matter_0)(r_max_0)

title = "$\Omega_m^{true}$=%.2f\n$\Omega_m^{fiducial}}$=%.2f\n$l_{max}$=%d, $k_{min}$=%.1f, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $R$=%.3f, $\overline{n}$=%.1f, $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_min, k_max, r_max_0, z_max, R, nbar, n_max)


# Plot and analyse the log likelihood function

plotLikelihoodResults(omega_matters, logLikelihoods, omega_matter_true, title)

# %%