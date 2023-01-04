# %%
import numpy as np

from utils import calculate_n_max_l
from generate_f_lmn import generate_f_lmn
from precompute_c_ln import load_c_ln_values
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from compute_likelihood import calc_all_Ws_without_delta_omega_m, computeLikelihood

from distance_redshift_relation import *


l_max = 20
# k_max = 10
k_max = 300
r_max = 0.8
n_max = calculate_n_max_l(0, k_max, r_max) # There are the most modes when l=0


c_ln_values = load_c_ln_values("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")
f_lmn = generate_f_lmn(l_max, n_max, r_max)


# %%

# Set a "true" cosmology
omega_matter = 0.5

# Assume a fiducial cosmology
omega_matter_0 = 0.5



# %%

# omega_matters = np.linspace(0.45, 0.55, 11)
omega_matters = np.linspace(0.485, 0.515, 46)
# omega_matters = np.linspace(0.48, 0.52, 61)
# omega_matters = np.linspace(0.4, 0.6, 61)
# omega_matters = np.linspace(0.43, 0.57, 15)
# omega_matters = np.linspace(0.4, 0.6, 21)

omega_matters

# %%

dr_domega = getPartialRbyOmegaMatterInterp(omega_matter_0)
Ws_without_delta_omega_m = calc_all_Ws_without_delta_omega_m(l_max, k_max, r_max, dr_domega)

# %%

likelihoods = [computeLikelihood(f_lmn, k_max, r_max, omega_m, omega_matter_0, Ws_without_delta_omega_m) for omega_m in omega_matters]

# %%

# Calculate the redshift limit equivalent to the radial limit
z_max = getInterpolatedZofR(omega_matter)(r_max)

plt.plot(omega_matters, likelihoods)
# plt.plot(omega_matters, likelihoods, '.')
plt.xlabel("$\Omega_m$")
plt.ylabel("ln L")
plt.title("ln L($\Omega_m$)\n$\Omega_m^{true}$=%.2f\n$l_{max}$=%d, $k_{max}$=%.1f, $r_{max}$=%.2f ($z_{max}$=%.2f), $n_{max,0}$=%d" % (omega_matter, l_max, k_max, r_max, z_max, n_max))
plt.show()

# %%

# Find the maximum
peak_index = np.argmax(likelihoods)
print("Peak is at Ωₘ = %.4f" % omega_matters[peak_index])

# Find the index of the true Ωₘ
true_index = np.argmin(np.abs(omega_matters - omega_matter))

print("ln L(true Ωₘ) = %.3f" % np.real(likelihoods[true_index]))
print("ln L(peak Ωₘ) = %.3f" % np.real(likelihoods[peak_index]))
print("ln L(true Ωₘ) - ln L(peak Ωₘ) = %.3f" % np.real(likelihoods[true_index] - likelihoods[peak_index]))
print("L(true Ωₘ) / L(peak Ωₘ) = %.3e" % np.exp(np.real(likelihoods[true_index] - likelihoods[peak_index])))


# %%
