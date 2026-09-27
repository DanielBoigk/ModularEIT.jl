---
tags: [diffusion, sde]
aliases: [Reverse SDE, Anderson's theorem]
---

**Theorem (Anderson 1982).** If $x(t)$ solves the forward SDE

$$
\mathrm dx = f(x,t)\,\mathrm dt+g(t)\,\mathrm dW_t
$$

with marginal densities $p_t$, then the time-reversed process solves

$$
\mathrm dx = \big[f(x,t)-g(t)^2\,\nabla_x\log p_t(x)\big]\mathrm dt+g(t)\,\mathrm d\bar W_t ,
$$

where time runs *backwards* from $T$ to $0$ ($\mathrm dt<0$) and $\bar W$ is a Brownian motion in reverse time. Starting from $x(T)\sim p_T$ and integrating to $t = 0$ produces samples of $p_0 = p_{\text{data}}$.

**For the [[Variance-Preserving SDE]]** ($f = -\frac12\beta x$, $g = \sqrt\beta$) with the learned score $s_\theta = -\varepsilon_\theta/\sqrt{1-\bar\alpha}$:

$$
\mathrm dx = \Big[-\tfrac12\beta(t)\,x+\frac{\beta(t)}{\sqrt{1-\bar\alpha(t)}}\,\varepsilon_\theta(x,t)\Big]\mathrm dt+\sqrt{\beta(t)}\,\mathrm d\bar W_t .
$$

The score term pulls the sample towards the data distribution. Because $\mathrm dt<0$, the backward step moves *against* the predicted noise.

It is discretised with the [[Euler-Maruyama Method]] or with DDPM's exact Gaussian step (see [[DDPM Ancestral Sampling]]). The deterministic counterpart with the same marginals is the [[Probability Flow ODE]].

## References

1. B. D. O. Anderson (1982). *Reverse-time diffusion equation models*. Stoch. Proc. Appl. 12(3), 313–326. [doi:10.1016/0304-4149(82)90051-5](https://doi.org/10.1016/0304-4149(82)90051-5)
2. Y. Song et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
