---
tags: [diffusion]
aliases: [Score, Stein score]
---

The **score** of a density $p$ is the gradient of its log:

$$
s(x) = \nabla_x\log p(x).
$$

It points towards regions of higher probability. The normalising constant is irrelevant, since $\nabla\log(p/Z) = \nabla\log p$, so scores can be learned for unnormalised models such as [[Energy-Based Models]].

**Scores of the noisy marginals.** Diffusion models need $\nabla_x\log p_t(x)$ for the marginal $p_t$ of the forward process at every time $t$. For the Gaussian transition kernel of the [[DDPM Forward Process]],

$$
\nabla_{x_t}\log q(x_t\mid x_0) = -\frac{x_t-\sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t} = -\frac{\varepsilon}{\sqrt{1-\bar\alpha_t}},
$$

using $x_t-\sqrt{\bar\alpha_t}x_0 = \sqrt{1-\bar\alpha_t}\,\varepsilon$.

**Noise prediction = score.** A network $\varepsilon_\theta(x_t,t)$ trained to predict the added noise (see [[Denoising Score Matching]]) gives the score estimate

$$
s_\theta(x_t,t) = -\frac{\varepsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}\approx\nabla_{x_t}\log p_t(x_t).
$$

Note that $\varepsilon_\theta$ estimates the *conditional expectation* $\mathbb E[\varepsilon\mid x_t]$. It therefore targets the score of the marginal $p_t$, not of the conditional $q(x_t\mid x_0)$.

**Why noise helps.** Real data lie near a low-dimensional set (see [[Manifold Hypothesis]]). The score of $p_{\text{data}}$ is undefined or unreliable off that set. Adding noise fills the ambient space, which gives well-defined scores everywhere and a continuum of noise levels to anneal through.

## References

1. A. Hyvärinen (2005). *Estimation of Non-Normalized Statistical Models by Score Matching*. J. Mach. Learn. Res. 6, 695–709. [jmlr.org/papers/v6/hyvarinen05a.html](https://jmlr.org/papers/v6/hyvarinen05a.html)
2. Y. Song, S. Ermon (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS 32. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
3. Y. Song et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
