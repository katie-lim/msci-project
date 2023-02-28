# %%
import numpy as np
from scipy.special import spherical_jn
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


# Selection function
def gaussianPhi(r, R):
    return np.exp(-r*r/(2*R*R))


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



def getZerosOfJ_lUpToBoundary(l, upperLimit):
    n = 0
    root = sphericalBesselZeros[l][0]

    if root < upperLimit:
        while root < upperLimit:
            n += 1
            root = sphericalBesselZeros[l][n]

        n_max = n - 1

        return sphericalBesselZeros[l][:n_max + 1]
        
    else:
        return []


def integrateWSplitByZeros(n, n_prime, l, r_max, r0OfR, rOfR0, phiOfR0, simpson=False, simpsonNpts=None, plot=False):
    k_ln = sphericalBesselZeros[l][n] / r_max
    k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max


    def W_integrand(r):
        r0 = r0OfR(r)

        return phiOfR0(r0) * spherical_jn(l, k_ln_prime*r) * spherical_jn(l, k_ln*r0) * r*r

    r_boundary = k_ln_prime * r_max
    r0_boundary = k_ln * r0OfR(r_max)

    r_zeros = getZerosOfJ_lUpToBoundary(l, r_boundary) / k_ln_prime
    r0_zeros = getZerosOfJ_lUpToBoundary(l, r0_boundary) / k_ln

    # Convert r0 values to r values
    r0_zeros = rOfR0(r0_zeros)

    # Combine and sort the zeros
    zeros = np.sort(np.append(r_zeros, r0_zeros))

    # Remove any duplicate zeros (which occur in the case r = r0)
    zeros = np.unique(zeros)


    zeros = np.append(zeros, [r_max])
    zeros = np.insert(zeros, 0, 0)


    if plot:
        x = np.linspace(0, r_max, 2000)
        y = W_integrand(x)

        plt.figure(dpi=200)
        plt.plot(x, y)
        plt.vlines(zeros, np.min(y), np.max(y), "r", "dotted")
        plt.show()


    integral = 0

    if simpson:
        for i in range(0, np.size(zeros) - 1):
            integral += computeIntegralSimpson(W_integrand, zeros[i], zeros[i+1], simpsonNpts)
    else:
        for i in range(0, np.size(zeros) - 1):
            integralChunk, error = quad(W_integrand, zeros[i], zeros[i+1])
            integral += integralChunk


    return np.power(r_max, -3/2) * c_ln_values_without_r_max[l][n_prime] * integral



def plotField(grid, r_i, r_max, k_max, l_max, lmax_calc):

    title = "r_i = %.2f, r_max = %.2f, k_max = %.2f, l_max = %d, lmax_calc = %d" % (r_i, r_max, k_max, l_max, lmax_calc)

    # Plot the field using the Mollweide projection

    fig = grid.plotgmt(projection='mollweide', colorbar='right', title=title)
    fig.show()