---
tags: [noise, modelling-error, bayesian]
aliases: [Approximation error, Bayesian approximation error approach, Enhanced error model, Modelling error]
---

Every reconstruction model is an approximation: a coarser mesh than reality, pixels instead of a continuous conductivity, idealised electrodes. The data, however, come from reality (or from a finer simulation, see [[Inverse Crime]]). If $F$ is the accurate forward model and $\tilde F$ the one used for the reconstruction,

$$
d = F(\sigma) + e = \tilde F(\sigma) + \varepsilon(\sigma) + e,\qquad \varepsilon(\sigma) = F(\sigma) - \tilde F(\sigma),
$$

with the **modelling error** $\varepsilon$ and the measurement noise $e\sim\mathcal N(0,\eta^2 I)$. When $\varepsilon$ is larger than the noise, fitting the data to the noise level means fitting the modelling error as well. The reconstruction then acquires artefacts, typically at the electrodes, where $\tilde F$ is least accurate, and the error grows again after an initial decrease (semi-convergence).

## The model (Kaipio and Somersalo)

The approximation error approach treats $\varepsilon$ as Gaussian, $\varepsilon\sim\mathcal N(\mu,\Gamma)$ independent of $\sigma$, and estimates $\mu$ and $\Gamma$ from samples $\sigma^{(i)}$ of the prior. Each sample is simulated with both models, and $\varepsilon^{(i)} = F(\sigma^{(i)}) - \tilde F(\sigma^{(i)})$. The misfit becomes

$$
\tfrac12\big\lVert C^{-1/2}\big(\tilde F(\sigma) + \mu - d\big)\big\rVert^2,\qquad C = \eta^2 I + \Gamma .
$$

The noise-level target for the whitened residual is again $\tfrac12\tau^2 m$ for $m$ residuals (see [[Choosing the Regularization Parameter]]). The model costs nothing during the reconstruction: all simulations with the accurate model happen beforehand.

## Few samples, many residuals

With $K$ samples, the sample covariance $\hat\Gamma$ has rank below $K$, often far below the number $m$ of residuals. A new modelling error then has components outside the span of the samples, which $\hat\Gamma$ assigns zero variance. In $C = \eta^2 I + \hat\Gamma$ they are divided by the noise level alone and dominate the whitened misfit, by orders of magnitude when the noise is small. A variance floor $\nu^2$ for the unseen directions fixes this, $C = (\eta^2+\nu^2)I + \hat\Gamma$.

$\nu$ can be estimated from the samples by leave-one-out: how much of each sample lies outside the span of the others, per remaining dimension. Because the span of the others is itself perturbed, this estimate is conservative, which is the safe side for whitening.

The whitening then needs no $m\times m$ matrix. With the thin SVD $Q S V^\top$ of the scaled sample deviations,

$$
C^{-1/2} = \sigma_d^{-1}\big(I - Q\operatorname{diag}(w_k)Q^\top\big),\qquad w_k = 1 - \frac{1}{\sqrt{1+s_k^2}},\qquad \sigma_d^2 = \eta^2+\nu^2 .
$$

Jacobians, Gram matrices and matrix-free products of the whitened residual follow from those of $\tilde F$ (see [[Gauss-Newton Method]]).

## In EIT

Nissinen, Heikkinen and Kaipio (2008) showed with measured data that the approach allows much coarser meshes and unknown boundary shapes. With many electrodes and low noise, the modelling error of any practical mesh exceeds the noise. In a simulated example with 255 electrodes and 1 % noise, the true conductivity misfits the noise-level target by a factor of eight. There, the approximation error model is not an option but a requirement: without it, the noise-level stopping rule overfits.

**In ModularEIT.jl:** [`ApproximationError`](https://danielboigk.github.io/ModularEIT.jl/dev/api/data/#ModularEIT.ApproximationError), [`ApproximationErrorObjective`](https://danielboigk.github.io/ModularEIT.jl/dev/api/data/#ModularEIT.ApproximationErrorObjective), [`whiten`](https://danielboigk.github.io/ModularEIT.jl/dev/api/data/#ModularEIT.whiten).

## References

1. J. Kaipio, E. Somersalo (2007). *Statistical inverse problems: Discretization, model reduction and inverse crimes*. J. Comput. Appl. Math. 198(2), 493–504. [doi:10.1016/j.cam.2005.09.027](https://doi.org/10.1016/j.cam.2005.09.027)
2. S. R. Arridge, J. P. Kaipio, V. Kolehmainen, M. Schweiger, E. Somersalo, T. Tarvainen, M. Vauhkonen (2006). *Approximation errors and model reduction with an application in optical diffusion tomography*. Inverse Problems 22(1), 175–195. [doi:10.1088/0266-5611/22/1/010](https://doi.org/10.1088/0266-5611/22/1/010)
3. A. Nissinen, L. M. Heikkinen, J. P. Kaipio (2008). *The Bayesian approximation error approach for electrical impedance tomography—experimental results*. Meas. Sci. Technol. 19(1), 015501. [doi:10.1088/0957-0233/19/1/015501](https://doi.org/10.1088/0957-0233/19/1/015501)
