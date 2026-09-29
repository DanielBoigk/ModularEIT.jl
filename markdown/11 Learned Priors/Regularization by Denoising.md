---
tags: [machine-learning, regularization]
aliases: [RED]
---

**Regularization by Denoising (RED)** builds an *explicit* regulariser from a denoiser $D$:

$$
\mathcal R_{\text{RED}}(x) = \tfrac12\,x^\top\big(x-D(x)\big).
$$

It is small when $x$ is close to its own denoised version, that is, when $x$ looks like a clean image.

**Gradient.** Romano, Elad and Milanfar showed that if $D$ is *locally homogeneous* ($D(cx) = cD(x)$ for $c$ near 1) and has a *symmetric Jacobian*, then

$$
\nabla\mathcal R_{\text{RED}}(x) = x-D(x) ,
$$

so gradient-based solvers only need one denoiser evaluation per step and no backpropagation through $D$. Reehorst and Schniter (2019) showed that practical denoisers violate Jacobian symmetry, so $x-D(x)$ is generally *not* the gradient of any function. RED algorithms are better understood as finding fixed points $x$ with $\nabla\Phi_{\text{data}}(x)+\beta(x-D(x)) = 0$ (score-matching interpretation).

**Connection to scores.** For a MMSE Gaussian denoiser with noise level $s$, [[Tweedie's Formula]] gives $D(x)-x = s^2\nabla\log p_s(x)$. The RED step $x-D(x)$ is therefore a scaled negative score. [[RED-Diff]] extends this to a whole range of noise levels using a diffusion model.

## References

1. Y. Romano, M. Elad, P. Milanfar (2017). *The Little Engine That Could: Regularization by Denoising (RED)*. SIAM J. Imaging Sci. 10(4), 1804–1844. [doi:10.1137/16M1102884](https://doi.org/10.1137/16M1102884)
2. E. T. Reehorst, P. Schniter (2019). *Regularization by Denoising: Clarifications and New Interpretations*. IEEE Trans. Comput. Imaging 5(1), 52–67. [doi:10.1109/TCI.2018.2880326](https://doi.org/10.1109/TCI.2018.2880326)
