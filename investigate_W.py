# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import spherical_jn

from utils import calc_n_max_l
from generate_f_lmn import p

# Full integral (no Taylor expansion)
from compute_likelihood_selection_func_shot_noise_no_tayl_exp import calc_W

# Taylor expansion
from compute_likelihood_selection_func_shot_noise import calc_W_1st_term, calc_W_2nd_term_without_delta_omega_m

from distance_redshift_relation import *


# %%

omega_matter_true = 0.5
omega_matter_0 = 0.5
l_max = 15
k_max = 300
r_max_true = 0.75
R = 0.25


radii_true = np.linspace(0, r_max_true, 1000)
true_z_of_r = getInterpolatedZofR(omega_matter_true)
z_true = true_z_of_r(radii_true)

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)
r_max_0 = radii_fiducial[-1]

# %%

# Investigate the form of W_nn'^l

n, n_prime, l = 60, 59, 15

omega_matters = np.linspace(omega_matter_0 - 0.02, omega_matter_0 + 0.02, 51)
W_vals_full = []
W_vals_tayl = []

# Precompute parts of W for when we use the Taylor expansion
W_1st_term = calc_W_1st_term(n, n_prime, l, r_max_0, R)
dr_domega = getPartialRbyOmegaMatterInterp(omega_matter_0)
W_2nd_term_without_delta_omega_m = calc_W_2nd_term_without_delta_omega_m(n, n_prime, l, r_max_0, R, dr_domega)


for omega_matter in omega_matters:
    # Do the full integral (no Taylor expansion)
    r0OfR = getInterpolatedR0ofR(omega_matter_0, omega_matter)

    W_val_full = calc_W(n, n_prime, l, r_max_0, R, r0OfR)
    W_vals_full.append(W_val_full)


    # Now calculate W using the Taylor expansion
    W_val_tayl = W_1st_term + W_2nd_term_without_delta_omega_m * (omega_matter_0 - omega_matter)
    W_vals_tayl.append(W_val_tayl)
    


# Compare the two methods

plt.figure(dpi=200)
plt.plot(omega_matters, W_vals_full, label="Full integral")
plt.plot(omega_matters, W_vals_tayl, label="Taylor expansion")
plt.xlabel("$\Omega_m$")
plt.title("$W_{%d,%d}^{%d}$" % (n, n_prime, l))
plt.legend()
plt.show()

# %%
