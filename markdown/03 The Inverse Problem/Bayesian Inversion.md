---
tags: [inverse-problem, statistics]
aliases: [Statistical inversion]
---

In the **Bayesian** (statistical) approach, the unknown conductivity $\sigma$ and the data $y$ are random variables. With a prior density $\pi(\sigma)$ and a likelihood $\pi(y\mid\sigma)$, Bayes' rule gives the **posterior**

$$
\pi(\sigma\mid y) \propto \pi(y\mid\sigma)\,\pi(\sigma).
$$

For additive Gaussian noise $y = \mathcal F(\sigma) + \eta$, $\eta\sim\mathcal N(0,\Gamma)$:

$$
-\log\pi(\sigma\mid y) = \tfrac12\|y-\mathcal F(\sigma)\|_{\Gamma^{-1}}^2 - \log\pi(\sigma) + \text{const}.
$$

**Connection to regularisation.** The *maximum a posteriori* (MAP) estimate minimises exactly a [[Variational Regularization|variational objective]]: the data misfit plus $\mathcal R(\sigma) = -\log\pi(\sigma)$. For example, a Gaussian smoothness prior gives [[Tikhonov Regularization]], and a Laplace-type prior on gradients gives [[Total Variation]].

**Beyond the MAP.** Markov chain Monte Carlo (MCMC) sampling of the posterior gives the conditional mean, credible intervals and uncertainty quantification. Kaipio et al. (2000) did this for EIT. It is expensive because every sample needs a forward solve. Generative priors such as [[Diffusion Models]] are the learned counterpart of $\pi(\sigma)$ (see [[Diffusion Posterior Sampling]]). Linearizing at the MAP estimate gives a Gaussian approximation of the posterior, with the pointwise variance available cheaply from the Jacobian's SVD (see [[Resolution and Confidence Maps]]).

## References

1. J. P. Kaipio, V. Kolehmainen, E. Somersalo, M. Vauhkonen (2000). *Statistical inversion and Monte Carlo sampling methods in electrical impedance tomography*. Inverse Problems 16(5), 1487–1522. [doi:10.1088/0266-5611/16/5/321](https://doi.org/10.1088/0266-5611/16/5/321)
2. J. Kaipio, E. Somersalo (2005). *Statistical and Computational Inverse Problems*. Springer. [doi:10.1007/b138659](https://doi.org/10.1007/b138659)
