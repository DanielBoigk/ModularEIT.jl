---
tags: [diffusion, sde]
aliases: [VP-SDE]
---

The continuous-time limit of the [[DDPM Forward Process]] is the **variance-preserving SDE**

$$
\mathrm dx = -\tfrac12\beta(t)\,x\,\mathrm dt+\sqrt{\beta(t)}\,\mathrm dW_t ,
$$

a time-inhomogeneous Ornstein–Uhlenbeck process with drift $f(x,t) = -\frac12\beta(t)x$ and diffusion coefficient $g(t) = \sqrt{\beta(t)}$.

**Explicit solution.** With $A(t) = \int_0^t\beta(s)\,\mathrm ds$, the integrating factor $e^{A(t)/2}$ gives

$$
x(t) = e^{-A(t)/2}x(0)+\int_0^te^{-(A(t)-A(s))/2}\sqrt{\beta(s)}\,\mathrm dW_s .
$$

The stochastic integral is Gaussian with mean zero and, by the Itô isometry, covariance

$$
\int_0^te^{-(A(t)-A(s))}\beta(s)\,\mathrm ds\;I = \big(1-e^{-A(t)}\big)I .
$$

Hence

$$
x(t)\mid x(0)\sim\mathcal N\big(\sqrt{\bar\alpha(t)}\,x(0),\ (1-\bar\alpha(t))I\big),\qquad\bar\alpha(t) = e^{-A(t)},
$$

which matches the discrete marginals exactly. Squared mean scale plus variance equals $\bar\alpha+(1-\bar\alpha) = 1$, which is where the name *variance preserving* comes from.

Song et al. also define the *variance-exploding* SDE $\mathrm dx = \sqrt{\mathrm d[\sigma^2(t)]/\mathrm dt}\,\mathrm dW$, the continuous version of noise-conditional score networks. Reversing either SDE requires the [[Score Function]] (see [[Reverse-Time SDE]]).

## References

1. Y. Song et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
2. B. Øksendal (2003). *Stochastic Differential Equations*, 6th ed. Springer. [doi:10.1007/978-3-642-14394-6](https://doi.org/10.1007/978-3-642-14394-6)
