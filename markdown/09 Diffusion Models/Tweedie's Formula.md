---
tags: [diffusion, statistics]
aliases: [Tweedie, Denoised estimate, x0 prediction]
---

**Tweedie's formula.** If $y = \mu+s\,\eta$ with $\eta\sim\mathcal N(0,I)$ and $y$ has marginal density $p$, then the posterior mean of the clean signal is

$$
\mathbb E[\mu\mid y] = y+s^2\,\nabla_y\log p(y).
$$

The MMSE denoiser only needs the score of the *noisy* marginal. The formula goes back to Robbins (1956), who credited Tweedie; Efron (2011) gives a modern account.

**For diffusion models.** With $x_t = \sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$, apply the formula to $\mu = \sqrt{\bar\alpha_t}x_0$ and $s^2 = 1-\bar\alpha_t$:

$$
\hat x_0(x_t) := \mathbb E[x_0\mid x_t] = \frac{1}{\sqrt{\bar\alpha_t}}\Big(x_t+(1-\bar\alpha_t)\,\nabla_{x_t}\log p_t(x_t)\Big).
$$

With the learned score $s_\theta = -\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$:

$$
\hat x_0(x_t)\approx\frac{1}{\sqrt{\bar\alpha_t}}\Big(x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon_\theta(x_t,t)\Big).
$$

This is the forward formula solved for $x_0$ with the predicted noise.

**Role in inverse problems.** A data-fidelity term $\|y-\mathcal A(x)\|$ only makes sense for *clean* images, but during sampling only noisy $x_t$ is available. Tweedie gives a clean estimate in closed form at every step, so the likelihood can be evaluated at $\hat x_0(x_t)$:

- as a gradient with respect to $x_t$, backpropagating through $\hat x_0$ ([[Diffusion Posterior Sampling]]);
- as the starting point of a data-consistency prox ([[DiffPIR]]).

$\hat x_0$ is a posterior *mean*. It is blurry at high noise levels, when many clean images are compatible with $x_t$, and sharp at low noise levels.

## References

1. B. Efron (2011). *Tweedie's Formula and Selection Bias*. J. Amer. Statist. Assoc. 106(496), 1602–1614. [doi:10.1198/jasa.2011.tm11181](https://doi.org/10.1198/jasa.2011.tm11181)
2. H. Robbins (1956). *An Empirical Bayes Approach to Statistics*. Proc. Third Berkeley Symp. Math. Stat. Prob. 1, 157–163. [projecteuclid.org](https://projecteuclid.org/euclid.bsmsp/1200501653)
3. H. Chung, J. Kim, M. T. McCann, M. L. Klasky, J. C. Ye (2023). *Diffusion Posterior Sampling for General Noisy Inverse Problems*. ICLR 2023. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)
