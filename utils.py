# %%
import numpy as np
from scipy.integrate import quad, simpson
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
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



# The default error tolerance used by scipy.quad is epsabs=1.49e-8
def computeIntegralSplit(integrand, N, upperLimit, epsabs=1.49e-8):
    answer = 0
    step = upperLimit / N

    for i in range(N):
        integral, error = quad(integrand, i*step, (i+1)*step, epsabs=epsabs)
        answer += integral
        # print(error)

    return answer



def computeIntegralSimpson(integrand, lowerLimit, upperLimit, Npts):

    x = np.linspace(lowerLimit, upperLimit, Npts)
    y = integrand(x)

    integral = simpson(y, dx=x[1] - x[0])

    return integral






def plotField(grid, r_i, r_max, k_max, l_max, lmax_calc):

    title = "r_i = %.2f, r_max = %.2f, k_max = %.2f, l_max = %d, lmax_calc = %d" % (r_i, r_max, k_max, l_max, lmax_calc)

    # Plot the field using the Mollweide projection

    fig = grid.plotgmt(projection='mollweide', colorbar='right', title=title)
    fig.show()