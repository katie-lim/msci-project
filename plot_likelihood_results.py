# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def plotLikelihoodResults(omega_matters, logLikelihoods, omega_matter_true, title):
    
    # Convert from complex numbers to floats
    logLikelihoods = np.real(logLikelihoods)


    # Find the maximum of the likelihood
    peak_index = np.argmax(logLikelihoods)
    omega_m_peak = omega_matters[peak_index]

    # Find the index of the true Ωₘ
    true_index = np.argmin(np.abs(omega_matters - omega_matter_true))

    print("Peak Ωₘ = %.4f" % omega_m_peak)
    print("ln L(true Ωₘ) = %.3f" % np.real(logLikelihoods[true_index]))
    print("ln L(peak Ωₘ) = %.3f" % np.real(logLikelihoods[peak_index]))
    print("ln L(true Ωₘ) - ln L(peak Ωₘ) = %.3f" % np.real(logLikelihoods[true_index] - logLikelihoods[peak_index]))
    print("L(true Ωₘ) / L(peak Ωₘ) = %.3e" % np.exp(np.real(logLikelihoods[true_index] - logLikelihoods[peak_index])))


    # Plot the likelihood
    lnL_peak = logLikelihoods[peak_index]
    delta_lnL = logLikelihoods - lnL_peak

    plt.figure(dpi=200)
    plt.plot(omega_matters, np.exp(delta_lnL))
    plt.xlabel("$\Omega_m$")
    plt.ylabel("L/L$_{peak}$")
    plt.title("L($\Omega_m$)/L$_{peak}$\n%s" % title)
    plt.show()


    # Estimate the width, sigma, by fitting a quadratic to the log likelihood

    def quadratic(x, sigma):
        return -1/2 * ((x - omega_m_peak)/sigma)**2


    params, cov = curve_fit(quadratic, omega_matters, delta_lnL, [1])
    sigma = np.abs(params[0])


    # Plot the log likelihood function, with the fit overlaid
    plt.figure(dpi=200)
    plt.plot(omega_matters, delta_lnL, ".", label="$\Delta$ ln L")
    plt.plot(omega_matters, quadratic(omega_matters, *params), label="Gaussian fit")

    plt.xlabel("$\Omega_m$")
    plt.ylabel("$\Delta$ ln L")
    plt.title("$\Delta$ ln L($\Omega_m$)\n%s" % title)
    plt.legend()
    plt.show()


    print("Result: Ωₘ = %.5f +/- %.5f" % (omega_m_peak, sigma))
    # %%
