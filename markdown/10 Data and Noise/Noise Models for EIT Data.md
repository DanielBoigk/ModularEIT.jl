---
tags: [data, noise]
---

Simulated EIT data must be corrupted with noise to mimic real measurements and to avoid overly optimistic results (see also [[Inverse Crime]]).

**Additive Gaussian noise on voltages** (most common). For each pattern:

$$
f_i^\delta = f_i+\eta_i,\qquad \eta_i\sim\mathcal N\big(0,\ (\delta\,\|f_i\|/\sqrt m)^2 I\big),
$$

relative to the signal level ($m$ is the number of boundary values), or with a fixed absolute standard deviation from the instrument specification.

**Noise on currents and voltages separately.** With parameters $\sigma_g$ (current-source error) and $\sigma_f$ (voltmeter error):

$$
f_i = \mathcal R_\gamma\big(g_i+\sigma_g\,\varepsilon_{g,i}\big)+\sigma_f\,\varepsilon_{f,i},\qquad \varepsilon\sim\mathcal N(0,I).
$$

The injected current is projected back to zero mean before the solve. The algorithm is given the nominal $g_i$ and the noisy $f_i$.

**Operator-level noise.** With an estimated [[Discrete Boundary Operator]] $\mathcal R$, add a random *symmetric* perturbation

$$
\tilde{\mathcal R} = \mathcal R+s\,\mathcal E,\qquad \mathcal E = \tfrac12(\hat{\mathcal E}+\hat{\mathcal E}^\top),\quad \hat{\mathcal E}_{jk}\sim\mathcal N(0,1)\ \text{i.i.d.}
$$

This preserves symmetry but not positive semidefiniteness: for large $s$, small eigenvalues can become negative. New data pairs are then extracted from $\tilde{\mathcal R}$. Inverting $\tilde{\mathcal R}$ to get DtN data amplifies the noise strongly in the small-eigenvalue directions, giving heavy-tailed errors.

**Beyond white noise.** Real data also contain electrode contact-impedance errors, electrode position errors, drift and correlated noise. These *modelling errors* are often larger than instrument noise. The approximation error approach models them statistically.

**In ModularEIT.jl:** [`GaussianNoise`](https://danielboigk.github.io/ModularEIT.jl/dev/api/data/#ModularEIT.GaussianNoise), [`RelativeGaussianNoise`](https://danielboigk.github.io/ModularEIT.jl/dev/api/data/#ModularEIT.RelativeGaussianNoise), [`SourceMeterNoise`](https://danielboigk.github.io/ModularEIT.jl/dev/api/data/#ModularEIT.SourceMeterNoise), [`add_noise`](https://danielboigk.github.io/ModularEIT.jl/dev/api/data/#ModularEIT.add_noise), [`perturb_boundary_operator`](https://danielboigk.github.io/ModularEIT.jl/dev/api/data/#ModularEIT.perturb_boundary_operator), [`perturb_contact_impedance`](https://danielboigk.github.io/ModularEIT.jl/dev/api/data/#ModularEIT.perturb_contact_impedance), [`electrode_angles`](https://danielboigk.github.io/ModularEIT.jl/dev/api/data/#ModularEIT.electrode_angles).

## References

1. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM, p. 197 ff. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
2. J. Kaipio, E. Somersalo (2005). *Statistical and Computational Inverse Problems*. Springer. [doi:10.1007/b138659](https://doi.org/10.1007/b138659)
