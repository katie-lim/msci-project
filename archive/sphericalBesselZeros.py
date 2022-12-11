# %%
from precompute_sph_bessel_zeros import loadSphericalBesselZeros


zeros = loadSphericalBesselZeros("zeros.csv")


def getZerosUpTokMax(l, k_max, r_max):

    k_values = []
    k_val = 0
    n = 0

    while k_val < k_max:
        root = zeros[l][n]
        k_val = root/r_max

        k_values.append(k_val)
        n += 1

    return k_values


# %%


# getZerosUpTokMax(1, 50, 2)


# %%
