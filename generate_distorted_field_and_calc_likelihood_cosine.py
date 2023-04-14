# %%
import numpy as np
from scipy.optimize import curve_fit
from os import path

from utils import calc_n_max_l
from generate_field import generateTrueField, multiplyFieldBySelectionFunction
from spherical_bessel_transform import calc_f_lmn_0
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from compute_likelihood_selection_func_shot_noise_no_tayl_exp import calc_all_W, computeLikelihood

from distance_redshift_relation import *


l_max = 15
k_min = 0
k_max = 200
r_max_true = 0.75
n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0
# nbar = 5


c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

# %%

# First, generate a true field

omega_matter_true = 0.315
radii_true = np.linspace(0, r_max_true, 1001)

true_z_of_r = getInterpolatedZofR(omega_matter_true)
z_true = true_z_of_r(radii_true)

z_true, all_grids = generateTrueField(radii_true, omega_matter_true, r_max_true, l_max, k_max)

# %%

# Define a selection function with the form of a cosine
# That goes to 0 at the boundary

phi_r_max = r_max_true

def phi(r):
    return ((np.cos(r * np.pi/phi_r_max) + 1)/2) * (r < phi_r_max)

plt.plot(np.linspace(0, 5, 100), phi(np.linspace(0, 5, 100)))
plt.xlabel("r")
plt.ylabel("$\phi$(r)")
plt.show()

# %%

# Add the effect of the selection function

radii_true, all_observed_grids = multiplyFieldBySelectionFunction(radii_true, all_grids, phi)

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

# Or, load f_lmn_0 from a file
omega_matter_true = 0.315
omega_matter_0 = 0.315
l_max = 15
k_max = 200
r_max_true = 0.75
R = 0.25

saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_with_phi.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R)

f_lmn_0 = np.load(saveFileName)


# %%

# Calculate likelihood

omega_matters = np.linspace(omega_matter_0 - 0.01, omega_matter_0 + 0.01, 21)
# omega_matters = np.array([0.315])
likelihoods = []

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

#%%

for omega_matter in omega_matters:

    W_saveFileName = "data/W_tayl_exp_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    # W_saveFileName = "data/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    W = np.load(W_saveFileName)

    likelihood = computeLikelihood(f_lmn_0, k_min, k_max, r_max_0, omega_matter, W, nbar=1)
    likelihoods.append(likelihood)

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
plt.title("ln L($\Omega_m$)\n$\Omega_m^{true}$=%.2f\n$\Omega_m^{fiducial}}$=%.2f\n$l_{max}$=%d, $k_{min}$=%.1f, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $R$=%.3f, $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_min, k_max, r_max_0, z_max, R, n_max))
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
plt.title("L($\Omega_m$)/L$_{peak}$\n$\Omega_m^{true}$=%.2f\n$\Omega_m^{fiducial}}$=%.2f\n$l_{max}$=%d, $k_{min}$=%.1f, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $R$=%.3f, $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_min, k_max, r_max_0, z_max, R, n_max))
plt.show()

# %%

# Estimate the width, sigma

def quadratic(x, mean, sigma):
    return -1/2 * ((x - mean)/sigma)**2


params, cov = curve_fit(quadratic, omega_matters, delta_lnL, [omega_m_peak, 1])
sigma = np.abs(params[1])

print("σ = %.5f" % sigma)

# %%

plt.figure(dpi=400)
plt.plot(omega_matters, delta_lnL, ".", label="$\Delta$ ln L", c="#001848")
plt.plot(omega_matters, quadratic(omega_matters, *params), label="Gaussian fit", c="#70AD47", zorder=0)

plt.vlines(params[0], np.min(delta_lnL), np.max(delta_lnL), "b", "dotted")
# plt.text(params[0], (np.min(delta_lnL) + np.max(delta_lnL))/2, "peak", c="b")
# plt.text(params[0], 10, "$\Omega_m^{peak}$ = %.4f" % params[0], c="b")
plt.text(params[0] - 0.001, 10, "$\Omega_m^{peak}$ = %.4f" % params[0], c="b")


plt.vlines(omega_matter_true, np.min(delta_lnL), quadratic(omega_matter_true, *params), "r", "dotted")
plt.text(omega_matter_true - 0.006, 3, "$\Omega_m^{true}$ = 0.3150", c="r")



plt.ylim(top=30)
plt.xlabel("$\Omega_m$")
plt.ylabel("$\Delta$ ln L")
# plt.title("$\Delta$ ln L($\Omega_m$)\n$\Omega_m^{true}$=%.2f\n$\Omega_m^{fiducial}}$=%.2f\n$l_{max}$=%d, $k_{min}$=%.1f, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $R$=%.3f, $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_min, k_max, r_max_0, z_max, R, n_max))
# plt.title("$\Delta$ ln L($\Omega_m$)\n$\Omega_m^{true}$=%.3f\n$\Omega_m^{fiducial}}$=%.3f\n$l_{max}$=%d, $k_{min}$=%.1f, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $R$=%.3f, $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_min, k_max, r_max_0, z_max, R, n_max))
plt.title("$\Delta$ ln L($\Omega_m$)")
plt.legend(loc="lower left")
plt.show()


print("Result: Ωₘ = %.5f +/- %.5f" % (params[0], sigma))
# %%

# %%
from matplotlib import font_manager
sorted(font_manager.get_font_names())
# %%
# %%
np.save("x.npy", omega_matters)
# %%

np.save("y.npy", likelihoods)
# %%
omega_matters = np.load("x.npy")
likelihoods = np.load("y.npy")

# %%

import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = 'arial.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams.update({'font.size': 16})

# %%
plt.plot([1, 2, 3], [1, 2, 3])
# %%
