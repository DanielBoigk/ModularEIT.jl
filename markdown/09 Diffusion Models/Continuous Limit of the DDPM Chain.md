---
tags: [diffusion, sde]
---

Why does the discrete chain $x_i = \sqrt{1-\beta_i}\,x_{i-1}+\sqrt{\beta_i}\,\varepsilon_i$ converge to the [[Variance-Preserving SDE]]?

**Scaling.** Place $N$ steps on $[0,1]$ with $\Delta t = 1/N$ and set $\beta_i = \beta(t_i)\,\Delta t$ for a fixed continuous function $\beta$. The per-step noise variance must be of order $\Delta t$: only then does the accumulated noise converge to Brownian motion. Order $1$ would blow up, and order $\Delta t^2$ would vanish.

**Euler–Maruyama form.** Since $\sqrt{1-\beta\Delta t} = 1-\tfrac12\beta\Delta t+\mathcal O(\Delta t^2)$,

$$
x(t+\Delta t)-x(t) = -\tfrac12\beta(t)\,x(t)\,\Delta t+\sqrt{\beta(t)}\,\sqrt{\Delta t}\,\varepsilon+\mathcal O(\Delta t^2),
$$

which is one [[Euler-Maruyama Method|Euler–Maruyama]] step of $\mathrm dx = -\frac12\beta x\,\mathrm dt+\sqrt\beta\,\mathrm dW$.

**Convergence.** The first two conditional moments per unit time converge, $\frac1{\Delta t}\mathbb E[\Delta x\mid x]\to-\frac12\beta x$ and $\frac1{\Delta t}\mathbb E[\Delta x\,\Delta x^\top\mid x]\to\beta I$, and the fourth moment is $\mathcal O(\Delta t^2)$, which excludes jumps. The drift is Lipschitz, so the limiting martingale problem is well posed. The classical diffusion-limit theorem for Markov chains (Stroock–Varadhan; Ethier–Kurtz) then gives weak convergence of the interpolated chain to the SDE.

**Cumulative factor.** Taking logarithms turns the product into a sum:

$$
\log\bar\alpha_N(t) = \sum_{t_j\le t}\log\big(1-\beta(t_j)\Delta t\big) = -\sum_{t_j\le t}\beta(t_j)\Delta t+\mathcal O(\Delta t)\ \xrightarrow{N\to\infty}\ -\int_0^t\beta(s)\,\mathrm ds ,
$$

so $\bar\alpha(t) = \exp(-\int_0^t\beta)$. This is the formula used in the [[Noise Schedule]].

**Why not differentiate the marginal formula?** Differentiating $x(t) = \sqrt{\bar\alpha}x_0+\sqrt{1-\bar\alpha}\,\varepsilon$ in $t$ with $\varepsilon$ *fixed* gives a deterministic ODE. After replacing $\varepsilon$ by its conditional expectation it becomes the [[Probability Flow ODE]], which has the same marginals as the SDE but different paths. Marginals alone do not determine a diffusion; the Markov-chain limit selects the SDE.

## References

1. Y. Song et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021, App. B. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
2. S. N. Ethier, T. G. Kurtz (1986). *Markov Processes: Characterization and Convergence*. Wiley. [doi:10.1002/9780470316658](https://doi.org/10.1002/9780470316658)
3. D. W. Stroock, S. R. S. Varadhan (1979). *Multidimensional Diffusion Processes*. Springer (reprinted in Classics in Mathematics). [doi:10.1007/3-540-28999-2](https://doi.org/10.1007/3-540-28999-2)
