---
tags: [diffusion, ode]
---

For every forward SDE $\mathrm dx = f\,\mathrm dt+g\,\mathrm dW$ there is a deterministic ODE with the **same marginals** $p_t$:

$$
\frac{\mathrm dx}{\mathrm dt} = f(x,t)-\tfrac12g(t)^2\,\nabla_x\log p_t(x).
$$

For the [[Variance-Preserving SDE]]:

$$
\frac{\mathrm dx}{\mathrm dt} = -\tfrac12\beta(t)\big(x+\nabla_x\log p_t(x)\big) = -\tfrac12\beta(t)\Big(x-\frac{\varepsilon_\theta(x,t)}{\sqrt{1-\bar\alpha(t)}}\Big).
$$

**Derivation sketch.** Differentiating $x(t) = \sqrt{\bar\alpha}\,x_0+\sqrt{1-\bar\alpha}\,\varepsilon$ with $\varepsilon$ fixed, and using $\dot{\bar\alpha} = -\beta\bar\alpha$, gives $\dot x = -\frac12\beta x+\frac{\beta}{2\sqrt{1-\bar\alpha}}\varepsilon$. Replacing $\varepsilon$ by $\mathbb E[\varepsilon\mid x_t] = -\sqrt{1-\bar\alpha}\,\nabla\log p_t$ yields the ODE. Both the Fokker–Planck equation of the SDE and the continuity equation of the ODE are solved by the same $p_t$.

**Properties.**

- Deterministic: a bijection between noise and data. It enables exact likelihoods (as a continuous normalising flow; see [[Neural ODEs]]), latent interpolation, and fast high-order ODE solvers (DDIM is a discretisation of it).
- Same marginals, different paths: the SDE and the ODE agree at each time but not as path measures. A diffusion is not determined by its marginals.

## References

1. Y. Song et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
2. J. Song, C. Meng, S. Ermon (2021). *Denoising Diffusion Implicit Models*. ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
