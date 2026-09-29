---
tags: [diffusion, inverse-problems, regularization]
aliases: [RED-diff, Variational diffusion regularization]
---

**RED-Diff** (Mardani et al.) treats a pretrained diffusion model as a *variational regulariser* instead of a sampler. It does not integrate the reverse SDE and does not need the schedule to be followed step by step.

**Objective.** Approximate the posterior by $q(x_0\mid y) = \mathcal N(\mu,s^2I)$, in practice with $s\to0$, and minimise over $\mu$:

$$
\ell(\mu) = \frac{\|y-\mathcal A(\mu)\|^2}{2\sigma_y^2}+\lambda\,\mathbb E_{t,\varepsilon}\Big[\omega(t)\,\big\|\varepsilon_\theta\big(\sqrt{\bar\alpha_t}\,\mu+\sqrt{1-\bar\alpha_t}\,\varepsilon,\ t\big)-\varepsilon\big\|^2\Big].
$$

The regulariser asks how well the model can denoise noisy versions of $\mu$. It is small when $\mu$ looks like training data. It is a weighted sum of [[Denoising Score Matching]] losses over noise levels and relates to a KL divergence between $q$ and the diffusion prior.

**Stochastic gradient.** Following score distillation, the network Jacobian is dropped (stop-gradient):

$$
\nabla_\mu\ell\approx-\frac{1}{\sigma_y^2}\mathcal A'(\mu)^\top\big(y-\mathcal A(\mu)\big)+\lambda_t\,\big(\varepsilon_\theta(x_t,t)-\varepsilon\big),\qquad x_t = \sqrt{\bar\alpha_t}\,\mu+\sqrt{1-\bar\alpha_t}\,\varepsilon,
$$

with one random $(t,\varepsilon)$ per step and $\lambda_t$ absorbing the weighting. The regulariser gradient is a [[Regularization by Denoising|RED]]-type residual at a random noise level. The authors propose weights proportional to the inverse square root of the SNR, $\lambda_t\propto\sqrt{1-\bar\alpha_t}/\sqrt{\bar\alpha_t}$, and visiting $t$ in *descending* order, coarse to fine.

```text
μ ← μ0
for k = 1, …, K:
    t ~ schedule (descending or uniform),  ε ~ N(0, I)
    x_t ← sqrt(ᾱ_t) μ + sqrt(1−ᾱ_t) ε
    g_fit ← ∇ data misfit at μ                      (adjoint solve for EIT)
    g_reg ← λ_t (ε_θ(x_t, t) − ε)                    (no backprop through ε_θ)
    μ ← optimiser step (SGD / Adam) with g_fit + λ g_reg
return μ
```

**Properties.**

- Optimisation, not sampling. It gives a MAP-like point estimate with a stochastic gradient, so [[Stochastic and Adaptive Gradient Methods]] apply.
- No backpropagation through the network and no fixed step schedule. Evaluations at many $(t,\varepsilon)$ can be batched on a GPU.
- The regulariser can be wrapped into a [[Diffusion Proximal Operator]] for use in [[ADMM]].
- Being MAP-like, it gives no posterior samples. The authors note this and the dependence on the weighting as limitations.

## References

1. M. Mardani, J. Song, J. Kautz, A. Vahdat (2024). *A Variational Perspective on Solving Inverse Problems with Diffusion Models*. ICLR 2024. [arXiv:2305.04391](https://arxiv.org/abs/2305.04391)
2. B. Poole, A. Jain, J. T. Barron, B. Mildenhall (2023). *DreamFusion: Text-to-3D using 2D Diffusion*. ICLR 2023. [arXiv:2209.14988](https://arxiv.org/abs/2209.14988)
3. Y. Romano, M. Elad, P. Milanfar (2017). *The Little Engine That Could: Regularization by Denoising (RED)*. SIAM J. Imaging Sci. 10(4), 1804–1844. [doi:10.1137/16M1102884](https://doi.org/10.1137/16M1102884)
