
- Consider a field $f(r,\theta,\phi)$, representing the true number density of galaxies.
- Let the *fractional* fluctuations of this true field be $\delta(\mathbf{r})=\frac{f(\mathbf{r})-\bar{f}}{\bar{f}}$.
	- This implies $f(\mathbf{r}) = \bar{f}(1+\delta(\mathbf{r}))$.
- However, observations from galaxy surveys do not produce the true number density field $f(r,\theta,\phi)$.
- Instead, they observe galaxies out to a sphere defined by some redshift limit, $z_{max}$.
- Also, the observed field is modified by the *selection function* $\phi(z)$ – fainter galaxies are less likely to be observed.
- Let the observed field be $n(z,\theta,\phi)=\phi(z)f(z,\theta,\phi)$.
- Equivalently, this is equal to $n(r_{0}, \theta,\phi)=\phi(r_{0})f(r_{0},\theta,\phi)$, since we can convert redshifts to distances assuming a fiducial cosmology.


---

## Spherical Bessel transform

The expansion coefficients of the *true fractional fluctuations* are given by

$$
\delta_{lmn} = c_{l n}  \int _{0}^{r_{max}} \delta(\mathbf{r}) j_{l}(k_{l n}r) Y_{lm}^{*}(\theta, \phi) d^{3}\mathbf{r}.
$$
where $r_{max} = r(z_{max};\Omega_{m})$. Here $\Omega_{m}$ is the true value.

These obey $\langle \delta_{lmn}\delta_{l'm'n'}^{*} \rangle=P(k_{ln})\delta^{K}_{ll'}\delta^{K}_{mm'}\delta^{K}_{nn'}$ – i.e. in the absence of distortion, there is no coupling between any of the modes. The universe is **isotropic**, so there are factors of $\delta_{l l'}^{K}\delta_{m m'}^{K}$.

However, galaxy surveys do not produce the true field or the true fractional fluctuations. The expansion coefficients of the observed field $n(r_{0},\theta,\phi)$ are

$$
n_{lmn} = c_{l n} \int _{0}^{r_{max}^{0}} \phi(r_{0})f(\mathbf{r}_{0}) j_{l}(k_{l n}r_{0}) Y_{lm}^{*}(\theta, \phi) d^{3}\mathbf{r}_{0},
$$
where $r_{max}^{0} = r(z_{max};\Omega_{m}^{0})$.

Use the fact that the number of galaxies is conserved: $f(\mathbf{r}_{0})d^{3}\mathbf{r}_{0} = f(\mathbf{r}) d^{3}\mathbf{r}$.

$$
n_{lmn} = c_{l n} \int _{0}^{r_{max}} \phi(r_{0})f(\mathbf{r}) j_{l}(k_{l n}r_{0}) Y_{lm}^{*}(\theta, \phi) d^{3}\mathbf{r}
$$

Let's now write $f(\mathbf{r})$ in terms of $\delta(\mathbf{r})$, so that we're able to use the relation $\langle \delta_{lmn}\delta_{l'm'n'}^{*} \rangle=P(k_{ln})\delta^{K}_{ll'}\delta^{K}_{mm'}\delta^{K}_{nn'}$ later on.

$$
n_{lmn} = c_{l n} \int _{0}^{r_{max}} \phi(r_{0})\bar{f} (1+\delta(\mathbf{r}))  j_{l}(k_{l n}r_{0}) Y_{lm}^{*}(\theta, \phi) d^{3}\mathbf{r}
$$

We will neglect the $1$ in the factor of $(1+\delta(\mathbf{r}))$ because it is a constant, and therefore only affects the monopole mode ($l=m=0$). 

We thus omit the $1$ in our theory to simplify the maths. In our analysis, we will then need to omit the monopole mode $l=m=0$.

$$
n_{lmn} = \bar{f} c_{l n} \int _{0}^{r_{max}} \phi(r_{0}) \delta(\mathbf{r})  j_{l}(k_{l n}r_{0}) Y_{lm}^{*}(\theta, \phi) d^{3}\mathbf{r}
\tag{1}
$$

Now insert the expansion of the true fluctuation field $\delta(\mathbf{r})$
$$
\delta(\mathbf{r}) = \sum_{lmn} \delta_{lmn} c_{l n}j_{l}(k_{l n}r) Y_{lm}(\theta,\phi),
$$
back into (1). We obtain

$$
n_{lmn} = \bar{f} c_{l n} \int _{0}^{r_{max}} \phi(r_{0})\left[ \sum_{l'm'n'} \delta_{l'm'n'} c_{l' n'}j_{l'}(k_{l' n'}r) Y_{l'm'}(\theta,\phi) \right] j_{l}(k_{l n}r_{0}) Y_{lm}^{*}(\theta, \phi) r^{2}dr d\Omega
$$

Now perform the angular integral. Using the fact that the spherical harmonics are orthonormal ($\int Y_{l'm'}(\theta,\phi)Y_{lm}^{*}(\theta,\phi)d\Omega=\delta^{K}_{ll'}\delta^{K}_{mm'}$), we get two Kronecker deltas. These kill off the sum over $l'$ and $m'$ (but the sum over $n'$ remains):

$$
n_{lmn} = \bar{f} c_{l n} \int _{0}^{r_{max}} \phi(r_{0})\left[ \sum_{n'} \delta_{lmn'} c_{l n'}j_{l}(k_{l n'}r) \right] j_{l}(k_{l n}r_{0}) ~ r^{2}dr
$$
or
$$
n_{lmn} = \bar{f} ~ \sum_{n'} \delta_{lmn'} W_{n n'}^{l}
$$
where
$$
W_{nn'}^{l} = c_{l n}c_{l n'} \int_{0}^{r_{max}} \phi(r_{0}) j_{l}(k_{l n'}r) j_{l}(k_{l n}r_{0}) \, r^{2}dr.
$$
---

## Expectation values

$$
\langle n_{lmn}n_{l'm'n'}^{*} \rangle = \bar{f}^{2} \left\langle  \left( \sum_{n''} \delta_{lmn''}W_{nn''}^{l} \right) \left(  \sum_{n'''} \delta_{l'm'n'''}^{*}W_{n'n'''}^{l'}  \right)  \right\rangle
$$
$$
= \bar{f}^{2}  \sum_{n''}\sum_{n'''} \langle \delta_{lmn''}\delta_{l'm'n'''}^{*} \rangle   W_{nn''}^{l} W_{n'n'''}^{l'}
$$
But $\langle \delta_{lmn}\delta_{l'm'n'}^{*} \rangle=P(k_{ln})\delta^{K}_{ll'}\delta^{K}_{mm'}\delta^{K}_{nn'}$, so this reduces to

$$
= \bar{f}^{2}  \sum_{n''}\sum_{n'''}  P(k_{ln''})\delta_{l l'}^{K} \delta_{m m'}^{K} \delta_{n''n'''}^{K} ~  W_{nn''}^{l} W_{n'n'''}^{l'}
$$
$$
= \bar{f}^{2}  \sum_{n''} P(k_{ln''})\delta_{l l'}^{K} \delta_{m m'}^{K} ~  W_{nn''}^{l} W_{n'n''}^{l'}
$$
$$
\boxed{
\langle n_{lmn}n_{l'm'n'}^{*} \rangle = \bar{f}^{2} \delta_{l l'}^{K} \delta_{m m'}^{K} \sum_{n''} P(k_{ln''})  W_{nn''}^{l} W_{n'n''}^{l'}
}
$$
