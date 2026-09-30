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

## Linearized data consistency for EIT

Solving the nonlinear prox at every diffusion step costs PDE solves. If a reconstruction $\theta_0$ that fits the data to the noise level is available, for example from [[Levenberg-Marquardt Method|Levenberg–Marquardt]], the residual can be linearised there once, $r(\theta)\approx r_0 + J(\theta-\theta_0)$. The prox then has a closed form through the SVD $J = U S V^\top$:

$$
\operatorname*{argmin}_\theta\ \frac{\lVert r_0 + J(\theta-\theta_0)\rVert^2}{\eta^2} + \frac{\lVert\theta-\hat\theta\rVert^2}{\gamma_t^2}
= \hat\theta - V\operatorname{diag}\Big(\frac{s_i}{s_i^2+\eta^2/\gamma_t^2}\Big)U^\top\big(r_0 + J(\hat\theta-\theta_0)\big),
$$

with the residual noise level $\eta$ and the trust $\gamma_t\propto\bar\sigma_t/\sqrt\lambda$ in the denoised estimate $\hat\theta$. Mode by mode, directions the data determine ($s_i\gg\eta/\gamma_t$) are taken from the data, and the others are left to the prior. This is the filter of the model resolution matrix (see [[Resolution and Confidence Maps]]), applied at every step, and the sampling needs no PDE solve. The linearised term knows no bounds, so the result is clipped to the range of the training images.

In the landscape example of EITDenoiser.jl (32 electrodes, 1 % noise, 64 × 64 pixels) the samples fit the *nonlinear* data at the noise level, agree where the data determine the conductivity (sky, height of the horizon, dark ground), and differ in the undetermined texture (haze, the outline of the ridge). Their mean is slightly more accurate than the Levenberg–Marquardt reconstruction the linearisation started from.

Where a sample misses the nonlinear data slightly, a few Levenberg–Marquardt steps close the gap. They must be *strongly* damped: then they move the sample by well under a percent. Weakly damped steps re-fit the data in poorly determined directions and nearly double the error, destroying what the prior contributed.

**In EITDenoiser.jl:** `LinearizedData`, `data_prox`, `pixel_consistency`, `diffusion_sample`, `polish_sample`, see the [repository](https://github.com/DanielBoigk/EITDenoiser.jl).

**Compared with DPS.** No backpropagation through the network is needed, and the data term is enforced by optimisation rather than by a single gradient step. This makes DiffPIR much less sensitive to the relative scaling of prior and likelihood and suits expensive nonlinear operators better. It is the diffusion analogue of [[Plug-and-Play Priors]].

## References

1. Y. Zhu, K. Zhang, J. Liang, J. Cao, B. Wen, R. Timofte, L. Van Gool (2023). *Denoising Diffusion Models for Plug-and-Play Image Restoration*. CVPR Workshops 2023. [doi:10.1109/CVPRW59228.2023.00129](https://doi.org/10.1109/CVPRW59228.2023.00129)
2. K. Zhang, Y. Li, W. Zuo, L. Zhang, L. Van Gool, R. Timofte (2021). *Plug-and-Play Image Restoration with Deep Denoiser Prior*. IEEE TPAMI. [arXiv:2008.13751](https://arxiv.org/abs/2008.13751)
