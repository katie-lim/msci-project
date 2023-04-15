# %%
import numpy as np
from os import path

from generate_field import generateTrueField, multiplyFieldBySelectionFunction
from distance_redshift_relation import *
from spherical_bessel_transform import calc_f_lmn_0
from compute_likelihood import calc_all_W, computeLikelihood
from analyse_likelihood import plotContour
from utils import calc_n_max_l, gaussianPhi
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros


l_max = 15
k_max = 200
r_max_true = 0.75
n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0
R = 0.25 # Selection function scale length
# nbar = 5


c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

# %%

# First, generate a true field

omega_matter_true = 0.315
radii_true = np.linspace(0, r_max_true, 1001)

true_z_of_r = getInterpolatedZofR(omega_matter_true)
z_true = true_z_of_r(radii_true)

# %%

z_true, all_grids = generateTrueField(radii_true, omega_matter_true, r_max_true, l_max, k_max)

# %%

# Add the effect of the selection function

def phi(r0):
    return gaussianPhi(r0, R)

radii_true, all_observed_grids = multiplyFieldBySelectionFunction(radii_true, all_grids, phi)

# %%

# --------------- OBSERVED

omega_matter_0 = 0.315

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)
r_max_0 = radii_fiducial[-1]

# %%

# Perform the spherical Bessel transform to obtain the coefficients

f_lmn_0 = calc_f_lmn_0(radii_fiducial, all_observed_grids, l_max, k_max, n_max)

# %%

# Or, load f_lmn_0 from a file
omega_matter_true = 0.315
omega_matter_0 = 0.315
l_max = 15
k_max = 200
r_max_true = 0.75
R = 0.25
P_amp = 1

saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_P-amp_%.2f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R, P_amp)

f_lmn_0 = np.load(saveFileName)


# %%

# Calculate likelihood

omega_matters = np.linspace(omega_matter_0 - 0.005, omega_matter_0 + 0.005, 11)
# P_amps = [1]
P_amps = np.linspace(0.95, 1.05, 5)
# omega_matters = np.array([0.315])
# likelihoods = []

likelihoods = np.zeros((np.size(omega_matters), np.size(P_amps)))

# %%

for omega_matter in omega_matters:

    W_saveFileName = "data/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)

    if path.exists(W_saveFileName):
        W = np.load(W_saveFileName)
    else:
        print("Computing W's for Ωₘ = %.3f." % omega_matter)
        r0OfR = getInterpolatedR0ofR(omega_matter_0, omega_matter)
        rOfR0 = getInterpolatedR0ofR(omega_matter, omega_matter_0)
        W = calc_all_W(l_max, k_max, r_max_0, R, r0OfR, rOfR0)
        np.save(W_saveFileName, W)

# %%

for i, omega_matter in enumerate(omega_matters):
    W_saveFileName = "data/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    W = np.load(W_saveFileName)

    for j, P_amp in enumerate(P_amps):
        print("Computing likelihood for Ωₘ = %.3f, P_amp = %.2f" % (omega_matter, P_amp))

        def p(k):
            if k < k_max:
                return P_amp
            else:
                return 0

        k_min = 0
        likelihood = computeLikelihood(f_lmn_0, k_min, k_max, r_max_0, W, p, nbar=1)
        likelihoods[i][j] = likelihood

# Convert from complex numbers to floats
# likelihoods = np.real(likelihoods)

# %%

# Calculate the redshift limit equivalent to the radial limit
# (assuming the fiducial cosmology)
z_max = getInterpolatedZofR(omega_matter_0)(r_max_0)

title = "$\Omega_m^{true}$=%.4f\n$\Omega_m^{fiducial}}$=%.4f\n$l_{max}$=%d, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $R$=%.3f, $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_0, z_max, R, n_max)

plotContour(omega_matters, P_amps, likelihoods, title)

# %%