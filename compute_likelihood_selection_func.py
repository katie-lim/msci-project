# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import spherical_jn

from utils import calc_n_max_l
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from generate_f_lmn import p


c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


# The default error tolerance used by scipy.quad is epsabs=1.49e-8
def computeIntegralSplit(integrand, N, upperLimit, epsabs=1.49e-8):
    answer = 0
    step = upperLimit / N

    for i in range(N):
        integral, error = quad(integrand, i*step, (i+1)*step, epsabs=epsabs)
        answer += integral
        # print(error)

    return answer



# Computes c_ln c_ln' \int_0^r_max j_l(k_ln' r) r^2 e^(-r^2 / 2R^2) j_l(k_ln r)
def calc_W_1st_term(n, n_prime, l, r_max, R, Nsplit=10, epsabs=1.49e-8):
    k_ln = sphericalBesselZeros[l][n] / r_max
    k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max


    def W_integrand(r):
        return r*r * spherical_jn(l, k_ln_prime*r) * np.exp(-r*r/(2*R*R)) * spherical_jn(l, k_ln*r)


    integral = computeIntegralSplit(W_integrand, Nsplit, r_max, epsabs)

    return np.power(r_max, -3) * c_ln_values_without_r_max[l][n] * c_ln_values_without_r_max[l][n_prime] * integral


# Computes the second term in the integral for W
def calc_W_2nd_term_without_delta_omega_m(n, n_prime, l, r_max, R, dr_domega, Nsplit=10, epsabs=1.49e-8):
    k_ln = sphericalBesselZeros[l][n] / r_max
    k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max


    def W_integrand(r):
        prefactor = r*r * spherical_jn(l, k_ln_prime*r) * np.exp(-r*r/(2*R*R)) * dr_domega(r)

        inner_brackets = (k_ln * spherical_jn(l, k_ln*r, True) - (r/(R*R)) * spherical_jn(l, k_ln*r))

        return prefactor * inner_brackets


    # integral, error = quad(integrand, 0, r_max)
    integral = computeIntegralSplit(W_integrand, Nsplit, r_max, epsabs)

    return np.power(r_max, -3) * c_ln_values_without_r_max[l][n] * c_ln_values_without_r_max[l][n_prime] * integral


def calc_all_W_1st_terms(l_max, k_max, r_max, R):
    # The maximum number of modes is when l=0
    n_max_0 = calc_n_max_l(0, k_max, r_max)

    W_lnn_prime_1st_term = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1))

    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max)

        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):
                W_lnn_prime_1st_term[l][n1][n2] = calc_W_1st_term(n1, n2, l, r_max, R)

    return W_lnn_prime_1st_term



def calc_all_W_2nd_terms_without_delta_omega_m(l_max, k_max, r_max, R, dr_domega):
    # The maximum number of modes is when l=0
    n_max_0 = calc_n_max_l(0, k_max, r_max)

    W_lnn_prime_2nd_term_without_delta_omega_m = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1))

    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max)

        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):
                W_lnn_prime_2nd_term_without_delta_omega_m[l][n1][n2] = calc_W_2nd_term_without_delta_omega_m(n1, n2, l, r_max, R, dr_domega)

    return W_lnn_prime_2nd_term_without_delta_omega_m




def computeExpectation(l, m, n, l_prime, m_prime, n_prime, k_max, r_max, omega_matter, omega_matter_0, P, W_1st_terms, W_2nd_terms_without_delta_omega_m):

    if (l == l_prime and m == m_prime):

        answer = 0

        delta_omega_matter = (omega_matter_0 - omega_matter)

        n_max_l = calc_n_max_l(l, k_max, r_max)

        for n_prime_prime in range(n_max_l + 1):
            k_ln_prime_prime = sphericalBesselZeros[l][n_prime_prime] / r_max

            W_n_nprimeprime = W_1st_terms[l][n][n_prime_prime] + (W_2nd_terms_without_delta_omega_m[l][n][n_prime_prime] * delta_omega_matter)
            W_nprime_nprimeprime = W_1st_terms[l][n_prime][n_prime_prime] + (W_2nd_terms_without_delta_omega_m[l][n_prime][n_prime_prime] * delta_omega_matter)

            answer += W_n_nprimeprime * np.conj(W_nprime_nprimeprime) * P(k_ln_prime_prime)

        return answer
    else:
        return 0



def computeLikelihood(f_lmn, k_max, r_max, omega_matter, omega_matter_0, W_1st_terms, W_2nd_terms_without_delta_omega_m):
    shape = f_lmn.shape
    l_max = shape[0] - 1 # -1 to account for l=0

    total = 0

    print("Computing likelihood for Ωₘ = %.3f" % omega_matter)


    for l in range(l_max + 1):
        # print("l =", l)
        n_max_l = calc_n_max_l(l, k_max, r_max)
        
        # Stop if there are no more modes
        if (n_max_l == -1): break


        # Construct the block of the covariance matrix for this l
        sigma_l = np.zeros((n_max_l + 1, n_max_l + 1))


        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):

                sigma_l[n1][n2] = computeExpectation(l, 0, n1, l, 0, n2, k_max, r_max, omega_matter, omega_matter_0, p, W_1st_terms, W_2nd_terms_without_delta_omega_m)
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
            # print("m =", m)
            for re_im in range(2):
                # For m = 0, the coeffs must be real
                if (m == 0) and (re_im == 1): continue


                # Construct the block of the data vector for this l, m, Re/Im
                data_block = []


                # Handle m = 0 separately
                # since for m = 0, the coefficients must be real
                if m == 0:
                    for n in range(n_max_l + 1):
                        data_block.append(f_lmn[l][m][n])
                else:
                    # Real block
                    if re_im == 0:
                        for n in range(n_max_l + 1):
                            data_block.append(np.real(f_lmn[l][m][n]))
                    # Imag block
                    else:
                        for n in range(n_max_l + 1):
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