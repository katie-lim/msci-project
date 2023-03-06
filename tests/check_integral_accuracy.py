# %%
import numpy as np
import matplotlib.pyplot as plt

from compute_Ws import calc_W_SplitIntegralByZeros, calc_W

from distance_redshift_relation import *


# %%

omega_matter_true = 0.315
omega_matter_0 = 0.315
l_max = 15
k_max = 200
r_max_true = 0.75


radii_true = np.linspace(0, r_max_true, 1000)
true_z_of_r = getInterpolatedZofR(omega_matter_true)
z_true = true_z_of_r(radii_true)

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)
r_max_0 = radii_fiducial[-1]

# %%

# Define a selection function with the form of a cosine
# That goes to 0 before the boundary

phi_r_max = 0.65

def phiOfR0(r0):
    return ((np.cos(r0 * np.pi/phi_r_max) + 1)/2) * (r0 < phi_r_max)

plt.plot(np.linspace(0, 5, 100), phiOfR0(np.linspace(0, 5, 100)))
plt.xlabel("$r_0$")
plt.ylabel("$\phi(r_0)$")
plt.show()

# %%

# Investigate the form of W_nn'^l

n, n_prime, l = 60, 60, 15
# n, n_prime, l = 20, 20, 5

omega_matters = np.linspace(omega_matter_0 - 0.02, omega_matter_0 + 0.02, 11)
W_vals_10_chunks = []
W_vals_1000_chunks = []
W_vals_full_zeros = []


for omega_matter in omega_matters:
    # Do the full integral (no Taylor expansion)
    r0OfR = getInterpolatedR0ofR(omega_matter_0, omega_matter)

    W_val_10_chunks = calc_W(n, n_prime, l, r_max_0, r0OfR, phiOfR0, Nsplit=10)
    W_vals_10_chunks.append(W_val_10_chunks)

    W_val_1000_chunks = calc_W(n, n_prime, l, r_max_0, r0OfR, phiOfR0, Nsplit=1000)
    W_vals_1000_chunks.append(W_val_1000_chunks)



    rOfR0 = getInterpolatedR0ofR(omega_matter, omega_matter_0)
    # print("omega_m = %.4f" % omega_matter)
    # W_val_full_zeros = integrateWSplitByZeros(n, n_prime, l, r_max_0, r0OfR, rOfR0, phi, simpson=True, simpsonNpts=1000)
    W_val_full_zeros = calc_W_SplitIntegralByZeros(n, n_prime, l, r_max_0, r0OfR, rOfR0, phiOfR0, plot=True)
    W_vals_full_zeros.append(W_val_full_zeros)



# %%
# Compare the two methods

plt.figure(dpi=200)
plt.plot(omega_matters, W_vals_10_chunks, label="Full integral (split into 10 chunks)")
plt.plot(omega_matters, W_vals_1000_chunks, label="Full integral (split into 1000 chunks)")
plt.plot(omega_matters, W_vals_full_zeros, label="Full integral (split by zeros)")
plt.xlabel("$\Omega_m$")
plt.title("$W_{%d,%d}^{%d}$" % (n, n_prime, l))
plt.legend()
plt.show()

# %%
