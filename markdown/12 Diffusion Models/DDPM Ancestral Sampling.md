---
tags: [diffusion, sampling]
aliases: [Reverse diffusion step, DDPM posterior]
---

For the discrete [[DDPM Forward Process]], the reverse step conditioned on a clean image is Gaussian with closed-form parameters:

$$
q(x_{t-1}\mid x_t,x_0) = \mathcal N\big(\tilde\mu_t(x_t,x_0),\ \tilde\beta_tI\big),
$$
$$
\tilde\mu_t = \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t+\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0,
\qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t .
$$

**Ancestral sampling.** $x_0$ is unknown during generation, so it is replaced by the denoised estimate from [[Tweedie's Formula]], $\hat x_0 = (x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon_\theta(x_t,t))/\sqrt{\bar\alpha_t}$. Then:

```text
x_T ← sample N(0, I)
for t = T, …, 1:
    ε̂  ← ε_θ(x_t, t)
    x̂0 ← (x_t − sqrt(1−ᾱ_t) ε̂) / sqrt(ᾱ_t)
    μ  ← sqrt(α_t)(1−ᾱ_{t−1})/(1−ᾱ_t) · x_t + sqrt(ᾱ_{t−1}) β_t/(1−ᾱ_t) · x̂0
    x_{t−1} ← μ + sqrt(β̃_t) · z,   z ~ N(0, I)   (z = 0 at t = 1)
return x_0
```

Substituting $\hat x_0$ gives the equivalent DDPM form $\mu = \frac1{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta\big)$. For small $\beta_t$ this step agrees with an [[Euler-Maruyama Method|Euler–Maruyama]] step of the [[Reverse-Time SDE]] to first order.

This step is the backbone that [[Diffusion Posterior Sampling]] augments with a likelihood gradient.

## References

1. J. Ho, A. Jain, P. Abbeel (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 33. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
