---
tags: [numerics, sde]
aliases: [Euler–Maruyama]
---

The **Euler–Maruyama** method is the simplest scheme for an SDE $\mathrm dx = f(x,t)\,\mathrm dt+g(t)\,\mathrm dW$:

$$
x_{n+1} = x_n+f(x_n,t_n)\,\Delta t+g(t_n)\sqrt{\Delta t}\,z_n,\qquad z_n\sim\mathcal N(0,I).
$$

The Brownian increment over $\Delta t$ has variance $\Delta t$, so the noise enters with $\sqrt{\Delta t}$. Convergence is of strong order $\tfrac12$ and weak order $1$.

**Reverse-time sampling.** For the [[Reverse-Time SDE]], stepping from $t$ to $t-\Delta t$ ($\Delta t>0$):

$$
x(t-\Delta t) = x(t)-\big[f(x,t)-g(t)^2\,s_\theta(x,t)\big]\Delta t+g(t)\sqrt{\Delta t}\,z .
$$

For the [[Variance-Preserving SDE]] with $s_\theta = -\varepsilon_\theta/\sqrt{1-\bar\alpha}$:

$$
x(t-\Delta t) = x(t)+\Big[\tfrac12\beta(t)\,x(t)-\frac{\beta(t)}{\sqrt{1-\bar\alpha(t)}}\,\varepsilon_\theta(x(t),t)\Big]\Delta t+\sqrt{\beta(t)\,\Delta t}\;z .
$$

The step slightly rescales $x$ up (undoing the contraction of the forward drift), removes a portion of the predicted noise, and injects fresh noise. At the final step the noise is usually omitted.

Predictor–corrector samplers alternate such a step with a few [[Langevin Dynamics|Langevin]] corrector steps at a fixed noise level.

## References

1. G. Maruyama (1955). *Continuous Markov processes and stochastic equations*. Rend. Circ. Mat. Palermo 4, 48–90. [doi:10.1007/BF02846028](https://doi.org/10.1007/BF02846028)
2. P. E. Kloeden, E. Platen (1992). *Numerical Solution of Stochastic Differential Equations*. Springer. [doi:10.1007/978-3-662-12616-5](https://doi.org/10.1007/978-3-662-12616-5)
3. Y. Song et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
