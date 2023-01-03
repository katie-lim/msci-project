# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import jv, spherical_jn

from precompute_c_ln import load_c_ln_values
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from distance_redshift_relation import getInterpolatedZofR, getPartialRbyOmegaMatterInterp
from generate_f_lmn import p


c_ln_values = load_c_ln_values("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


# omega_matter = 0.5
# omega_matter_0 = 0.2


def calculate_n_max_l(l, k_max, r_max):
    n = 0
    k_ln = sphericalBesselZeros[l][0] / r_max

    while k_ln < k_max:

        n += 1
        k_ln = sphericalBesselZeros[l][n] / r_max

    # if n == 0: return 0

    return n - 1


# The default error tolerance used by scipy.quad is epsabs=1.49e-8
def computeIntegralSplit(integrand, N, upperLimit, epsabs=1.49e-8):
    answer = 0
    step = upperLimit / N

    for i in range(N):
        integral, error = quad(integrand, i*step, (i+1)*step, epsabs=epsabs)
        answer += integral
        # print(error)

    return answer


# def calc_W(n, n_prime, l, r_max, omega_matter, omega_matter_0, dr_domega):
#     k_ln = sphericalBesselZeros[l][n] / r_max
#     k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max


#     def W_integrand(r):
#         return r*r * spherical_jn(l, k_ln_prime*r) * spherical_jn(l, k_ln*r, True) * dr_domega(r)


#     d_omega_matter = omega_matter_0 - omega_matter

#     # integral, error = quad(integrand, 0, r_max)
#     integral = computeIntegralSplit(W_integrand, 10, r_max)

#     return c_ln_values[l][n] * c_ln_values[l][n_prime] * k_ln * d_omega_matter * integral


def calc_W_without_delta_omega_m(n, n_prime, l, r_max, dr_domega, Nsplit=10, epsabs=1.49e-8, plotIntegrand=False):
    k_ln = sphericalBesselZeros[l][n] / r_max
    k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max


    def W_integrand(r):
        return r*r * spherical_jn(l, k_ln_prime*r) * spherical_jn(l, k_ln*r, True) * dr_domega(r)

    
    if plotIntegrand:
        r_vals = np.linspace(0, r_max, 500)
        plt.figure(dpi=200)
        plt.plot(r_vals, W_integrand(r_vals))
        plt.xlabel("r")
        plt.ylabel("integrand for W^%d_%d,%d" % (l, n, n_prime))
        plt.show()


    # integral, error = quad(integrand, 0, r_max)
    integral = computeIntegralSplit(W_integrand, Nsplit, r_max, epsabs)

    return c_ln_values[l][n] * c_ln_values[l][n_prime] * k_ln * integral



def calc_all_Ws_without_delta_omega_m(l_max, k_max, r_max, dr_domega):
    # The maximum number of modes is when l=0
    n_max_0 = calculate_n_max_l(0, k_max, r_max)

    W_lnn_prime = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1))

    for l in range(l_max + 1):
        n_max_l = calculate_n_max_l(l, k_max, r_max)

        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):
                W_lnn_prime[l][n1][n2] = calc_W_without_delta_omega_m(n1, n2, l, r_max, dr_domega)

    return W_lnn_prime




def computeExpectation(l, m, n, l_prime, m_prime, n_prime, k_max, r_max, omega_matter, omega_matter_0, P, Ws_without_delta_omega_m):

    if (l == l_prime and m == m_prime):

        k_ln = sphericalBesselZeros[l][n] / r_max
        k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max

        answer = 0

        if (n == n_prime):
            answer += P(k_ln)

        delta_omega_matter = (omega_matter_0 - omega_matter)

        W_nprime_n = Ws_without_delta_omega_m[l][n_prime][n] * delta_omega_matter
        W_n_nprime = Ws_without_delta_omega_m[l][n][n_prime] * delta_omega_matter


        n_max_l = calculate_n_max_l(l, k_max, r_max)
        n_prime_prime_sum = 0

        for n_prime_prime in range(n_max_l + 1):
            k_ln_prime_prime = sphericalBesselZeros[l][n_prime_prime] / r_max

            n_prime_prime_sum += (Ws_without_delta_omega_m[l][n][n_prime_prime] * delta_omega_matter) * np.conj(Ws_without_delta_omega_m[l][n_prime][n_prime_prime] * delta_omega_matter) * P(k_ln_prime_prime)


        answer += W_nprime_n * P(k_ln) + W_n_nprime * P(k_ln_prime) + n_prime_prime_sum

        return answer
    else:
        return 0



def computeLikelihood(f_lmn, k_max, r_max, omega_matter, omega_matter_0, Ws_without_delta_omega_m):
    shape = f_lmn.shape
    l_max = shape[0] - 1 # -1 to account for l=0

    total = 0

    print("Computing likelihood for Ωₘ = %.3f" % omega_matter)


    for l in range(l_max + 1):
        # print("l =", l)
        n_max_l = calculate_n_max_l(l, k_max, r_max)
        
        # Stop if there are no more modes
        if (n_max_l == -1): break


        # Construct the block of the covariance matrix for this l
        sigma_l = np.zeros((n_max_l + 1, n_max_l + 1))


        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):

                sigma_l[n1][n2] = computeExpectation(l, 0, n1, l, 0, n2, k_max, r_max, omega_matter, omega_matter_0, p, Ws_without_delta_omega_m)
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

    # likelihood = (1/np.sqrt(2 * np.pi * determinant)) * np.exp((-1/2) * total)

    # return likelihood



# %%

# calc_W(3, 4, 2, 1, 0.5, 0.2)

# %%

# from generate_f_lmn import p

# computeExpectation(2, 1, 3, 2, 1, 4, 1, 0.5, 0.2, p)


# %%

# f_lmn_0_loaded = np.load("f_lmn_0_values_15-11-2022.npy")
# l_max = 20
# k_max = 25
# r_max = 2.5
# n_max = 20
# omega_matter = 0.5
# omega_matter_0 = 0.2

# computeLikelihood(f_lmn_0_loaded, k_max, r_max, omega_matter, omega_matter_0)

# %%

# l = 1
# print("l =", l)
# n_max_l = calculate_n_max_l(l, k_max, r_max)

# # %%

# # Construct the block of the covariance matrix for this l
# sigma_l = np.zeros((n_max_l + 1, n_max_l + 1))

# # %%

# for n1 in range(n_max_l + 1):
#     for n2 in range(n_max_l + 1):
#         print(n1, n2)

#         sigma_l[n1][n2] = computeExpectation(l, 0, n1, l, 0, n2, k_max, r_max, omega_matter, omega_matter_0, p)

#         print(sigma_l[n1][n2])


# %%

# n1, n2 = 0, 2

# computeExpectation(l, 0, n1, l, 0, n2, k_max, r_max, omega_matter, omega_matter_0, p)

# %%


# calc_W(n2, n1, 2, r_max, omega_matter, omega_matter_0)
#* P(k_ln)

#+ calc_W(n, n_prime, l, r_max, omega_matter, omega_matter_0) * P(k_ln_prime)





# %%

# n = 1
# n_prime = 2

# k_ln = sphericalBesselZeros[l][n] / r_max
# k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max

# dr_domega = getPartialRbyOmegaMatterInterp(omega_matter_0)

# def W_integrand(r):
#     return r*r * spherical_jn(l, k_ln_prime*r) * spherical_jn(l, k_ln*r, True) * dr_domega(r)

# quad(W_integrand, 0, r_max)


# # %%
# import matplotlib.pyplot as plt

# x = np.linspace(0, r_max, 500)
# y = [W_integrand(xi) for xi in x]

# plt.plot(x, y)

# # %%

# quad(W_integrand, 0, r_max)


# %%
