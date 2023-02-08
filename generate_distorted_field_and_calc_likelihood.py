# %%
import numpy as np

from utils import calc_n_max_l
from generate_field import generateTrueField
from spherical_bessel_transform import calc_f_lmn_0
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from compute_likelihood import calc_all_Ws_without_delta_omega_m, computeLikelihood
from plot_likelihood_results import plotLikelihoodResults
from distance_redshift_relation import *


l_max = 15
k_max = 100
r_max_true = 0.8
n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0


c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

# %%

omega_matter_true = 0.5
radii_true = np.linspace(0, r_max_true, 1000)

z_true, all_grids = generateTrueField(radii_true, omega_matter_true, r_max_true, l_max, k_max)


# %%

# ----- OBSERVED


omega_matter_0 = 0.48

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)
r_max_0 = radii_fiducial[-1]


# %%

f_lmn_0 = calc_f_lmn_0(radii_fiducial, all_grids, l_max, k_max, n_max)

print(f_lmn_0)

# %%

saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true)

np.save(saveFileName, f_lmn_0)

# %%

# Or, load f_lmn_0 from a file
# f_lmn_0_loaded = np.load("f_lmn_0_true-0.5_fiducial-0.48_l_max-15_k_max-100_r_max_true-0.8.npy")
f_lmn_0_loaded = np.load("data/f_lmn_0_true-0.500_fiducial-0.480_l_max-15_k_max-100.00_r_max_true-0.800.npy")


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

omega_matters = np.linspace(omega_matter_0 - 0.03, omega_matter_0 + 0.03, 96)

# %%

dr_domega = getPartialRbyOmegaMatterInterp(omega_matter_0)
Ws_without_delta_omega_m = calc_all_Ws_without_delta_omega_m(l_max, k_max, r_max_0, dr_domega)


# %%

logLikelihoods = [computeLikelihood(f_lmn_0_loaded, k_max, r_max_0, omega_m, omega_matter_0, Ws_without_delta_omega_m) for omega_m in omega_matters]


# %%

# Calculate the redshift limit equivalent to the radial limit
# (assuming the fiducial cosmology)
z_max = getInterpolatedZofR(omega_matter_0)(r_max_0)

title = "$\Omega_m^{true}$=%.2f\n$\Omega_m^{fiducial}}$=%.2f\n$l_{max}$=%d, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_0, z_max, n_max)


# Plot and analyse the log likelihood function

plotLikelihoodResults(omega_matters, logLikelihoods, omega_matter_true, title)

# %%