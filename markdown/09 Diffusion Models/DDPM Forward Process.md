---
tags: [diffusion]
aliases: [Forward process, Noising process]
---

The **denoising diffusion probabilistic model (DDPM)** uses a Markov chain that gradually adds Gaussian noise to a data point $x_0$:

$$
x_t = \sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\varepsilon_t,\qquad\varepsilon_t\sim\mathcal N(0,I),\qquad t = 1,\dots,T,
$$

with a [[Noise Schedule]] $0<\beta_1<\dots<\beta_T<1$.

**Closed-form marginals.** With $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^t\alpha_s$, compounding the Gaussian steps gives

$$
q(x_t\mid x_0) = \mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big),
\qquad
x_t = \sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon .
$$

So any noise level can be sampled in one step, without simulating the chain:

```text
function ForwardSample(x0, t)
    ε  ← sample N(0, I)
    xt ← sqrt(ᾱ_t) · x0 + sqrt(1 − ᾱ_t) · ε
    return xt, ε
```

**Variance preserving.** If $\operatorname{Var}[x_0] = I$, then $\operatorname{Var}[x_t] = \bar\alpha_tI+(1-\bar\alpha_t)I = I$ for all $t$. The signal is replaced by noise without changing the overall scale. For $\bar\alpha_T\approx0$, $x_T$ is essentially standard Gaussian.

**Posterior of one step.** Given both $x_0$ and $x_t$, the previous state is Gaussian as well (see [[DDPM Ancestral Sampling]]). The continuous-time version of the chain is the [[Variance-Preserving SDE]] (see [[Continuous Limit of the DDPM Chain]]).

## References

1. J. Ho, A. Jain, P. Abbeel (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 33. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
2. J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, S. Ganguli (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML 2015. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585)
