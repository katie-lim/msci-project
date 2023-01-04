# %%
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


def calculate_n_max_l(l, k_max, r_max):
    n = 0
    k_ln = sphericalBesselZeros[l][0] / r_max

    while k_ln < k_max:

        n += 1
        k_ln = sphericalBesselZeros[l][n] / r_max

    # if n == 0: return 0

    return n - 1