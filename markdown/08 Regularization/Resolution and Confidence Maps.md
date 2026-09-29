---
tags: [regularization, uncertainty, spectral]
aliases: [Confidence map, Model resolution matrix, Linearized posterior variance, Sensitivity map]
---

A reconstruction does not determine every part of the conductivity equally well. Near the boundary and the electrodes the data pin the conductivity down; in the interior it is mostly the initial guess or the prior (see [[Decay of Boundary Measurements]]). A **confidence map** makes this visible pixel by pixel. All three maps below come from the Jacobian $J$ of the residual at the reconstruction, and from its SVD $J = U S V^\top W$ in a metric $W = \operatorname{diag}(w)$ on the parameters (see [[Truncated SVD Regularization]]).

## Sensitivity

The simplest map is the column norm $\|J e_j\|/w_j$: how much the data change when parameter $j$ changes (see [[Linearized EIT and the Sensitivity Kernel]]). It decays rapidly towards the interior. It ignores, however, that neighbouring parameters may affect the data in nearly the same way, and then cannot be told apart.

## Model resolution

A regularised linearized reconstruction recovers the perturbation $\delta\theta$ through the **model resolution matrix**

$$
\widehat{\delta\theta} = R\,\delta\theta,\qquad R = V F V^\top W,
$$

with filter factors $F = \operatorname{diag}(f_i)$. For truncation, $f_i = 1$ on the kept modes and $0$ otherwise. For Tikhonov or [[Levenberg-Marquardt Method|Levenberg–Marquardt]] damping in the metric $W$, $f_i = s_i^2/(s_i^2+\lambda)$. Since the columns of $W^{1/2}V$ are orthonormal, the diagonal

$$
R_{jj} = w_j\sum_i f_i\,V_{ji}^2 \in [0,1]
$$

is a confidence in the proper sense. A value of 1 means the data determine the parameter completely. A value of 0 means the reconstruction there is the initial guess. Unlike the sensitivity, the resolution accounts for the regularisation used, and it is not fooled by parameters the data cannot distinguish.

## Linearized posterior

With a Gaussian prior $\theta\sim\mathcal N(\theta_0,\gamma^2W^{-1})$ and residual noise $\mathcal N(0,\eta^2 I)$, the linearized posterior covariance (see [[Bayesian Inversion]]) is

$$
C = \bigl(J^\top J/\eta^2 + W/\gamma^2\bigr)^{-1},\qquad C_{jj} = \frac{\gamma^2}{w_j}\,\bigl(1 - R_{jj}\bigr),
$$

where $R$ is the Tikhonov resolution with $\lambda = \eta^2/\gamma^2$. The posterior variance is therefore the prior variance, reduced by the fraction the data resolve. With independent pixels as the prior, the pointwise reduction is moderate even near the boundary. The data determine combinations of neighbouring pixels well, but single pixels poorly, so the posterior standard deviation is best read in comparison across the domain rather than as an absolute error bar.

## Use

- **Diagnosis.** Features in regions of low resolution are not supported by the data. They come from the prior (or are artefacts), which is a warning for learned priors in particular (see [[Hallucinations and Uncertainty]]).
- **Weighting.** The resolution map weights data consistency against a prior in a combined reconstruction. Where the data are informative, the prior should change little. Where they are not, it may act freely (see [[Diffusion Posterior Sampling]]).

**In ModularEIT.jl:** [`sensitivity_map`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.sensitivity_map), [`resolution_map`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.resolution_map), [`posterior_std`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.posterior_std), [`jacobian_svd`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.jacobian_svd).

## References

1. R. C. Aster, B. Borchers, C. H. Thurber (2013). *Parameter Estimation and Inverse Problems*, 2nd ed. Academic Press. [doi:10.1016/C2009-0-61134-X](https://doi.org/10.1016/C2009-0-61134-X)
2. J. Kaipio, E. Somersalo (2005). *Statistical and Computational Inverse Problems*. Springer. [doi:10.1007/b138659](https://doi.org/10.1007/b138659)
3. P. C. Hansen (1987). *The truncated SVD as a method for regularization*. BIT 27, 534–553. [doi:10.1007/BF01937276](https://doi.org/10.1007/BF01937276)
