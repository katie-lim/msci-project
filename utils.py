# %%
import numpy as np
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


def calc_n_max_l(l, k_max, r_max):
    n = 0
    k_ln = sphericalBesselZeros[l][0] / r_max

    if k_ln < k_max:

        while k_ln < k_max:
            n += 1
            k_ln = sphericalBesselZeros[l][n] / r_max

        return n - 1
        
    else:
        return 0


# Selection function
def phi(r, R):
    return np.exp(-r*r/(2*R*R))


def plotField(grid, r_i, r_max, k_max, l_max, lmax_calc):

    title = "r_i = %.2f, r_max = %.2f, k_max = %.2f, l_max = %d, lmax_calc = %d" % (r_i, r_max, k_max, l_max, lmax_calc)

    # Plot the field using the Mollweide projection

    fig = grid.plotgmt(projection='mollweide', colorbar='right', title=title)
    fig.show()