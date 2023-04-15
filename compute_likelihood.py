# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn

from utils import calc_n_max_l, computeIntegralSplit, integrateWSplitByZeros
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from generate_f_lmn import p


c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")



def calc_W(n, n_prime, l, r_max, R, r0OfR, Nsplit=10, epsabs=1.49e-8):
    k_ln = sphericalBesselZeros[l][n] / r_max
    k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max


    def W_integrand(r):
        r0 = r0OfR(r)

        return r*r * np.exp(-r0*r0/(2*R*R)) * spherical_jn(l, k_ln_prime*r) * spherical_jn(l, k_ln*r0)


    integral = computeIntegralSplit(W_integrand, Nsplit, r_max, epsabs)

    return np.power(r_max, -3) * c_ln_values_without_r_max[l][n] * c_ln_values_without_r_max[l][n_prime] * integral


def calc_all_W(l_max, k_max, r_max, R, r0OfR, rOfR0, phiOfR0):
    # The maximum number of modes is when l=0
    n_max_0 = calc_n_max_l(0, k_max, r_max)

    W_lnn_prime = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1))


    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max)

        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):
                # W_lnn_prime[l][n1][n2] = calc_W(n1, n2, l, r_max, R, r0OfR)
                W_lnn_prime[l][n1][n2] = integrateWSplitByZeros(n1, n2, l, r_max, r0OfR, rOfR0, phiOfR0, simpson=True, simpsonNpts=1000)

    return W_lnn_prime



def computeExpectation(l, m, n, l_prime, m_prime, n_prime, k_max, r_max, P, W, nbar):

    answer = 0

    # Signal term
    if (l == l_prime and m == m_prime):

        n_max_l = calc_n_max_l(l, k_max, r_max)

        for n_prime_prime in range(n_max_l + 1):
            k_ln_prime_prime = sphericalBesselZeros[l][n_prime_prime] / r_max

            W_n_nprimeprime = W[l][n][n_prime_prime]
            W_nprime_nprimeprime = W[l][n_prime][n_prime_prime]

            answer += W_n_nprimeprime * np.conj(W_nprime_nprimeprime) * P(k_ln_prime_prime)


        # answer *= nbar*nbar


    # Shot noise term
    # answer += nbar * W[l][n][n_prime]


    return answer


def computeLikelihood(f_lmn, k_min, k_max, r_max, W, P, nbar):
    shape = f_lmn.shape
    l_max = shape[0] - 1 # -1 to account for l=0

    total = 0


    for l in range(l_max + 1):
        # print("l =", l)
        n_max_l = calc_n_max_l(l, k_max, r_max)
        n_min_l = calc_n_max_l(l, k_min, r_max)
        # print("l = %d, %d <= n <= %d" % (l, n_min_l, n_max_l))
        
        # Stop if there are no more modes
        if (n_max_l == -1): break


        # Construct the block of the covariance matrix for this l
        sigma_l = np.zeros((n_max_l + 1 - n_min_l, n_max_l + 1 - n_min_l))


        for n1 in range(n_min_l, n_max_l + 1):
            for n2 in range(n_min_l, n_max_l + 1):

                sigma_l[n1 - n_min_l][n2 - n_min_l] = computeExpectation(l, 0, n1, l, 0, n2, k_max, r_max, P, W, nbar)
                # Set l = l' and m = m' = 0 since the expectation does not vary with m


        # print("Σ_%d:" % l)
        # print(sigma_l)


        # Invert it
        sigma_l_inv = np.linalg.inv(sigma_l)

        # print("Σ_%d inverse:" % l)
        # print(sigma_l_inv)

        # Also include m != 0, where half the power goes into the real and imag components
        sigma_l_inv_half = np.linalg.inv(sigma_l/2)


        # Compute the determinant of Σ_l
        det_sigma_l = np.linalg.det(sigma_l)
        det_sigma_l_half = np.linalg.det(sigma_l/2)

        # print("det Σ_%d:" % l)
        # print(det_sigma_l)
        # print("det Σ_%d/2:" % l)
        # print(det_sigma_l_half)

        if det_sigma_l < 0 or det_sigma_l_half < 0:
            print("Determinant is negative:")
            print("det Σ_%d = %.3e" % (l, det_sigma_l))



        for m in range(l + 1):
            if (l == 0) and (m == 0): continue

            # print("m =", m)
            for re_im in range(2):
                # For m = 0, the coeffs must be real
                if (m == 0) and (re_im == 1): continue


                # Construct the block of the data vector for this l, m, Re/Im
                data_block = []


                # Handle m = 0 separately
                # since for m = 0, the coefficients must be real
                if m == 0:
                    for n in range(n_min_l, n_max_l + 1):
                        data_block.append(f_lmn[l][m][n])
                else:
                    # Real block
                    if re_im == 0:
                        for n in range(n_min_l, n_max_l + 1):
                            data_block.append(np.real(f_lmn[l][m][n]))
                    # Imag block
                    else:
                        for n in range(n_min_l, n_max_l + 1):
                            data_block.append(np.imag(f_lmn[l][m][n]))



                data_block = np.array(data_block)


                # For m = 0, all the power goes into the real component
                # For m != 0, half the power goes into the real component and the other half goes into the imag component
                # So we need to halve the expectation values for m != 0

                # Now perform the matrix multiplication for this block

                if m == 0:
                    total += np.matmul(np.transpose(data_block), np.matmul(sigma_l_inv, data_block))
                else:
                    total += np.matmul(np.transpose(data_block), np.matmul(sigma_l_inv_half, data_block))



                # Add contribution from determinant
                if m == 0:
                    total += np.log(det_sigma_l)
                else:
                    total += np.log(det_sigma_l_half)


    # print("Determinant:", determinant)
    # print("Total:", total)
    # print("Log determinant:", np.log(determinant))


    # lnL = -1/2 * np.log(2*np.pi*determinant) - (1/2) * total
    lnL = -1/2 * (np.log(2*np.pi) + total)

    # print("ln L:", lnL)

    return lnL



# %%