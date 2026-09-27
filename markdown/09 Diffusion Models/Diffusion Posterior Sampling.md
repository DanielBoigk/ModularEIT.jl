---
tags: [diffusion, inverse-problems]
aliases: [DPS]
---

**Diffusion Posterior Sampling (DPS)** samples from the posterior $p(x_0\mid y)$ of an inverse problem $y = \mathcal A(x_0)+\eta$ with a pretrained unconditional diffusion model. It needs no retraining.

**Posterior score.** By Bayes' rule,

$$
\nabla_{x_t}\log p_t(x_t\mid y) = \nabla_{x_t}\log p_t(x_t)+\nabla_{x_t}\log p_t(y\mid x_t).
$$

The first term is the learned [[Score Function]]. The likelihood of the noisy $x_t$ is intractable. DPS approximates it through the denoised estimate from [[Tweedie's Formula]]:

$$
p_t(y\mid x_t)\approx p\big(y\mid\hat x_0(x_t)\big)\quad\Rightarrow\quad
\nabla_{x_t}\log p_t(y\mid x_t)\approx-\frac1{2\sigma_y^2}\nabla_{x_t}\big\|y-\mathcal A(\hat x_0(x_t))\big\|^2 .
$$

The gradient is taken with respect to $x_t$ and therefore backpropagates through the denoiser network.

**Algorithm** (on top of [[DDPM Ancestral Sampling]]):

```text
x_T ← sample N(0, I)
for t = T, …, 1:
    ε̂   ← ε_θ(x_t, t)
    x̂0  ← (x_t − sqrt(1−ᾱ_t) ε̂) / sqrt(ᾱ_t)          (differentiable in x_t)
    x'  ← ancestral step: μ_t(x_t, x̂0) + sqrt(β̃_t) z
    x_{t−1} ← x' − ζ_t ∇_{x_t} ‖y − A(x̂0)‖²          (likelihood guidance)
return x_0
```

Chung et al. use the step size $\zeta_t = \zeta/\|y-\mathcal A(\hat x_0(x_t))\|$.

**Properties and pitfalls.**

- It works for nonlinear $\mathcal A$ (including a differentiable PDE solver), and both noise and prior are handled in one sampler.
- The step size $\zeta$ is delicate. Too small and the sample ignores the data. Too large and it leaves the noise manifold, producing artefacts or saturated images. The balance is especially hard when the data gradient has a very different scale from the score, as in EIT.
- Each step needs a forward solve *and* backpropagation through the denoiser, so it is expensive for PDE-based operators. The noisy early steps provide only coarse guidance.

## References

1. H. Chung, J. Kim, M. T. McCann, M. L. Klasky, J. C. Ye (2023). *Diffusion Posterior Sampling for General Noisy Inverse Problems*. ICLR 2023. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)
2. H. Chung, B. Sim, D. Ryu, J. C. Ye (2022). *Improving Diffusion Models for Inverse Problems using Manifold Constraints*. NeurIPS 35. [arXiv:2206.00941](https://arxiv.org/abs/2206.00941)
