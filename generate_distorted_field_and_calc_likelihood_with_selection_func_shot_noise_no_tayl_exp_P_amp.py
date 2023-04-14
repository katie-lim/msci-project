# %%
import numpy as np
from scipy.optimize import curve_fit
from os import path

from generate_field import generateTrueField, multiplyFieldBySelectionFunction
from utils import calc_n_max_l
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from compute_likelihood_selection_func_shot_noise_no_tayl_exp import calc_all_W, computeLikelihood

from distance_redshift_relation import *


l_max = 15
k_min = 0
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

import pyshtools as pysh
import matplotlib.pyplot as plt
import matplotlib as mpl

from cartopy import crs as ccrs

# %%

mpl.rcParams.update({"axes.grid" : True, "grid.color": "#333333"})

i = 500
# title = r"$\delta(\mathbf{r})$ at $r$=%.2f\n$r_{max}$=%.2f, $k_{max}$=%d, $l_{max}$=%d" % (radii_true[i], r_max_true, k_max, l_max)
title = r"$\delta(\mathbf{r})$ at $r$=%.2f" % radii_true[i] + "\n" + "$r_{max}$=%.2f, $k_{max}$=%d, $l_{max}$=%d" % (r_max_true, k_max, l_max)

fig, ax = all_grids[i].plot(
    projection=ccrs.Mollweide(),
    colorbar='right',
    cb_label=r'$\delta(r, \theta, \phi)$',
    title=title,
    grid=True,
    show=False)
plt.savefig("field.svg", transparent=True, dpi=300)
plt.show()

# r_max_true, k_max, l_max,

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
P_amps = np.linspace(0.95, 1.05, 11)
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

#%%

# for omega_matter in omega_matters:

#     W_saveFileName = "data/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
#     # W_saveFileName = "data/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
#     W = np.load(W_saveFileName)

#     likelihood = computeLikelihood(f_lmn_0, k_min, k_max, r_max_0, omega_matter, W, nbar=1)
#     likelihoods.append(likelihood)

# # Convert from complex numbers to floats
# likelihoods = np.real(likelihoods)

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

        likelihood = computeLikelihood(f_lmn_0, k_min, k_max, r_max_0, W, p, nbar=1)
        likelihoods[i][j] = likelihood

# Convert from complex numbers to floats
# likelihoods = np.real(likelihoods)

# %%

# Find the maximum

peak_index = np.argmax(likelihoods)


maximum = np.max(likelihoods)
delta_lnLs = likelihoods - maximum

# %%


import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)

# X, Y = np.meshgrid(x, y)

# Z1 = np.exp(-X**2 - Y**2)
# Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# Z = (Z1 - Z2) * 2

X, Y = np.meshgrid(omega_matters, P_amps)
Z = delta_lnLs

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title(r'$\Delta \ln L$')
plt.xlabel(r"$\Omega_m$")
plt.ylabel(r"$P_{amp}$")
plt.savefig("contour_plot.png", dpi=500)


# %%

plt.figure(dpi=400)
plt.imshow(delta_lnLs[:,3:8])
plt.colorbar()

# Where we want the ticks, in pixel locations
ticks_x = np.linspace(0,4,5)
ticks_y = np.linspace(0,10,11)
# What those pixel locations correspond to in data coordinates.
# Also set the float format here
# ticklabels_x = ["{:6.2f}".format(i) for i in np.exp(ticks_x/5)]
# ticklabels_y = ["{:6.2f}".format(i) for i in np.exp(ticks_y/5)]
ticklabels_x = ["%.1f" % P_amp for P_amp in P_amps[3:8]]
ticklabels_y = omega_matters

ax = plt.gca()
ax.set_xticks(ticks_x)
ax.set_xticklabels(ticklabels_x)
ax.set_yticks(ticks_y)
ax.set_yticklabels(ticklabels_y)

plt.xlabel(r"$P_{amp}$")
plt.ylabel(r"$\Omega_m$")
plt.title("$\Delta \ln L$")
plt.savefig("2d_likelihood.png", dpi=400)
plt.show()



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

p0 = [omega_m_peak, 0.001]
params, cov = curve_fit(quadratic, omega_matters, delta_lnL, p0)
sigma = np.abs(params[1])



# def quadratic(x, sigma):
#     return -1/2 * ((x - omega_m_peak)/sigma)**2

# p0 = [0.001]
# params, cov = curve_fit(quadratic, omega_matters, delta_lnL, p0)
# sigma = np.abs(params[0])


print("σ = %.5f" % sigma)

# %%

plt.figure(dpi=400)
plt.plot(omega_matters, delta_lnL, ".", label="$\Delta$ ln L", c="#000000")
# plt.plot(omega_matters, quadratic(omega_matters, *[omega_m_peak, 0.001]), label="Initial guess", c="#70AD47", zorder=0)

x = np.linspace(np.min(omega_matters), np.max(omega_matters), 100)
plt.plot(x, quadratic(x, *params), label="Gaussian fit", c="#73CF4F", zorder=0)
# "#6F84F9"
# #ADB9FC
# #314ff7

# plt.vlines(params[0], np.min(delta_lnL), np.max(delta_lnL), "b", "dotted")
# plt.text(params[0], (np.min(delta_lnL) + np.max(delta_lnL))/2, "peak", c="b")
# plt.text(params[0], 10, "$\Omega_m^{peak}$ = %.4f" % params[0], c="b")
# plt.text(params[0] - 0.001, 10, "$\Omega_m^{peak}$ = %.4f" % params[0], c="b")


ylim = -3.8
# plt.vlines(omega_matter_true, np.min(delta_lnL), quadratic(omega_matter_true, *params), "r", "dotted")
plt.vlines(omega_matter_true, ylim, 0, "#314ff7", "dashed")
plt.ylim(ylim)
# plt.text(omega_matter_true - 0.006, 3, "$\Omega_m^{true}$ = 0.3150", c="r")


# plt.ylim(top=30)
plt.xlabel("$\Omega_m$", fontsize=14)
# plt.ylabel("$\Delta$ ln L")
# plt.title("$\Delta$ ln L($\Omega_m$)\n$\Omega_m^{true}$=%.3f\n$\Omega_m^{fiducial}}$=%.3f\n$l_{max}$=%d, $k_{min}$=%.1f, $k_{max}$=%.1f, $r_{max}^0$=%.2f ($z_{max}$=%.2f), $R$=%.3f, $n_{max,0}$=%d" % (omega_matter_true, omega_matter_0, l_max, k_min, k_max, r_max_0, z_max, R, n_max))
plt.title("$\Delta$ ln L($\Omega_m$)", fontsize=16)
plt.legend(loc="lower left")
plt.savefig("lnL_1.svg")
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

omega_matters = omega_matters[:-1]
likelihoods = likelihoods[:-1]
# %%
