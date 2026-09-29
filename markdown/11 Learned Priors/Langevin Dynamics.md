---
tags: [machine-learning, sampling]
aliases: [SGLD, Unadjusted Langevin algorithm]
---

The **overdamped Langevin SDE**

$$
\mathrm dx = \nabla\log p(x)\,\mathrm dt+\sqrt2\,\mathrm dW_t
$$

has $p$ as its stationary distribution. Discretising with the Euler–Maruyama method (see [[Euler-Maruyama Method]]) gives the **unadjusted Langevin algorithm (ULA)**:

$$
x_{k+1} = x_k+\eta\,\nabla\log p(x_k)+\sqrt{2\eta}\,\xi_k,\qquad\xi_k\sim\mathcal N(0,I).
$$

For an energy-based model $p\propto e^{-E}$ this is $x_{k+1} = x_k-\eta\nabla E(x_k)+\sqrt{2\eta}\,\xi_k$: gradient descent plus correctly scaled noise. The noise scales like the *square root* of the step size.

- For small $\eta$ the chain samples approximately from $p$. The bias from discretisation is removed by a Metropolis correction (MALA).
- **Stochastic gradient Langevin dynamics (SGLD)** (Welling & Teh 2011) uses mini-batch gradients and decreasing step sizes, which turns an SGD optimiser into a posterior sampler.
- **Posterior sampling for inverse problems:** replace $\nabla\log p$ by $\nabla\log p(x\mid y) = \nabla\log p(y\mid x)+\nabla\log p(x)$, a data gradient plus a learned score. Annealed Langevin with decreasing noise levels was the first score-based generative sampler (Song & Ermon 2019).

## References

1. M. Welling, Y. W. Teh (2011). *Bayesian Learning via Stochastic Gradient Langevin Dynamics*. ICML 2011. [stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)
2. G. O. Roberts, R. L. Tweedie (1996). *Exponential convergence of Langevin distributions and their discrete approximations*. Bernoulli 2(4), 341–363. [doi:10.2307/3318418](https://doi.org/10.2307/3318418)
3. Y. Song, S. Ermon (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS 32. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
