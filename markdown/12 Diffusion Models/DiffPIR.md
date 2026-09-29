---
tags: [diffusion, inverse-problems, plug-and-play]
aliases: [Diff-PIR, Denoising diffusion plug-and-play image restoration]
---

**DiffPIR** (Zhu et al. 2023) integrates a diffusion model into a plug-and-play half-quadratic splitting scheme. At every step it (i) denoises with the diffusion model, (ii) enforces data consistency with a **proximal step**, and (iii) re-noises to the next noise level.

**Algorithm.** With $\bar\sigma_t = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$ and $\rho_t = \lambda\,\sigma_n^2/\bar\sigma_t^2$:

```text
x_T ← sample N(0, I)
for t = T, …, 1:
    x0_t ← (x_t − sqrt(1−ᾱ_t) ε_θ(x_t, t)) / sqrt(ᾱ_t)          Tweedie denoising
    x̂0   ← argmin_x ‖y − A(x)‖² + ρ_t ‖x − x0_t‖²               data-consistency prox
    ε̂    ← (x_t − sqrt(ᾱ_t) x̂0) / sqrt(1 − ᾱ_t)                  implied noise
    x_{t−1} ← sqrt(ᾱ_{t−1}) x̂0 + sqrt(1−ᾱ_{t−1}) (sqrt(1−ζ) ε̂ + sqrt(ζ) ε),  ε ~ N(0, I)
return x_0
```

- The prox is the [[Proximal Operator]] of the data term at the Tweedie estimate (see [[Tweedie's Formula]]). For EIT it is computed approximately with a few Gauss–Newton or [[L-BFGS-B]] iterations.
- $\rho_t$ grows as the noise decreases: the prior dominates early, the data late.
- $\zeta\in[0,1]$ mixes deterministic (DDIM-like, $\zeta = 0$) and stochastic re-noising.

**Compared with DPS.** No backpropagation through the network is needed, and the data term is enforced by optimisation rather than by a single gradient step. This makes DiffPIR much less sensitive to the relative scaling of prior and likelihood and suits expensive nonlinear operators better. It is the diffusion analogue of [[Plug-and-Play Priors]].

## References

1. Y. Zhu, K. Zhang, J. Liang, J. Cao, B. Wen, R. Timofte, L. Van Gool (2023). *Denoising Diffusion Models for Plug-and-Play Image Restoration*. CVPR Workshops 2023. [arXiv:2305.08995](https://arxiv.org/abs/2305.08995)
2. K. Zhang, Y. Li, W. Zuo, L. Zhang, L. Van Gool, R. Timofte (2021). *Plug-and-Play Image Restoration with Deep Denoiser Prior*. IEEE TPAMI. [arXiv:2008.13751](https://arxiv.org/abs/2008.13751)
