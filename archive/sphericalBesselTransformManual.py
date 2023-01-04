# %%
import numpy as np
from scipy.special import spherical_jn, sph_harm

from generate_f_lmn import generate_f_lmn
from precompute_c_ln import get_c_ln_values_with_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

# l_max = 10
# n_max = 10000
# k_max = 30
# r_max = 100
l_max = 3
n_max = 10000
k_max = 5
r_max = 10

c_ln_values = get_c_ln_values_with_r_max("c_ln.csv", r_max)
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")
f_lmn = generate_f_lmn(l_max, n_max)


def a_lm(r_i, l, m, k_max, r_max):
    s = 0

    k_ln = 0
    n = 0

    while k_ln < k_max:

        k_ln = sphericalBesselZeros[l][n] / r_max

        s += c_ln_values[l][n] * spherical_jn(l, k_ln * r_i) * f_lmn[l][m][n]

        n += 1

    return s


def inverseSphericalTransform(r_i, l_max, k_max, r_max):

    def f(theta, phi):

        s = 0
        for l in range(l_max):
            for m in range(-l, l + 1):

                if l == 0 and m == 0:
                    continue

                s += a_lm(r_i, l, m, k_max, r_max) * sph_harm(m, l, theta, phi)

                print("l, m:", l, m)
                print("contribution:", a_lm(r_i, l, m, k_max, r_max) * sph_harm(m, l, theta, phi))
                print("sph_harm:", sph_harm(m, l, theta, phi))
                print(s)

        return s


    return f

# %%

print(a_lm(100, 1, -1, k_max, r_max))

# %%


def createField(r, l_max, k_max):

    set_of_f = []

    for r_i in r:
        f_i = inverseSphericalTransform(r_i, l_max, k_max, r_max)

        set_of_f.append(f_i)

    return set_of_f


r_min, N_r_values = 1, 100
r = np.linspace(r_min, r_max, N_r_values)

# print(r)

# %%


fi = createField(r, l_max, k_max)
theta, phi = 1, 1.7

print(fi[0](theta, phi))


# %%
print(fi[1](theta, phi))
print(fi[5](theta, phi))
print(fi[10](theta, phi))

# %%
l, m = 1, 1

print(sph_harm(m, l, theta, phi))
print(sph_harm(-m, l, theta, phi))

# %%

print(sph_harm(l, m, theta, phi))
print(sph_harm(l, -m, theta, phi))

# %%
