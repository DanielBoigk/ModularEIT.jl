---
tags: [regularization]
aliases: [Regularization, Regularized reconstruction]
---

Because the [[Calderón Problem]] is severely ill-posed (see [[Stability of the Calderón Problem]]), EIT reconstructions trade exact data fit for stability. They minimise a **variational objective**

$$
\min_{\sigma\in\Sigma}\ \ \underbrace{d\big(\mathcal F(\sigma),\,y^\delta\big)}_{\text{data fidelity}} \;+\; \beta\,\underbrace{\mathcal R(\sigma)}_{\text{regulariser}} ,
$$

where

- $\mathcal F$ is the [[Forward Map]] (for example voltages for given [[Current Patterns]]) and $y^\delta$ the noisy data;
- $d$ is a distance or divergence (see [[Data Fidelity Terms]]), most often $\tfrac12\|\mathcal F(\sigma)-y^\delta\|_2^2$;
- $\mathcal R$ encodes prior knowledge: smoothness ([[Tikhonov Regularization]]), piecewise constancy ([[Total Variation]]), low rank or spectral truncation ([[Truncated SVD Regularization]]), or a learned prior ([[Learned Regularization]]);
- $\beta>0$ is the regularisation parameter (see [[Choosing the Regularization Parameter]]);
- $\Sigma$ is the admissible set, for example $\sigma_{\min}\le\sigma\le\sigma_{\max}$ (see [[Box Constraints on Conductivity]]).

**What makes it a regularisation method.** For linear problems, and under conditions for nonlinear ones, the minimisers converge to a true solution as the noise level $\delta\to0$, *provided $\beta=\beta(\delta)$ is chosen suitably* ($\beta\to0$ and $\delta^2/\beta\to0$). In the Bayesian view, the objective is the negative log-posterior and its minimiser is the MAP estimate (see [[Bayesian Inversion]]).

Regularisation can also be **implicit**: early stopping of iterative methods, restricting to few low-frequency current patterns, or a coarse discretisation (see [[Implicit Regularization]]).

The objective is minimised with gradient-based methods. Gradients come from the [[Adjoint State Method]], and minimisation uses [[Gauss-Newton Method|Gauss–Newton]], [[L-BFGS-B]] or splitting schemes such as [[ADMM]].

## References

1. H. W. Engl, M. Hanke, A. Neubauer (1996). *Regularization of Inverse Problems*. Kluwer. [doi:10.1007/978-94-009-1740-8](https://doi.org/10.1007/978-94-009-1740-8)
2. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
3. M. Benning, M. Burger (2018). *Modern regularization methods for inverse problems*. Acta Numerica 27, 1–111. [doi:10.1017/S0962492918000016](https://doi.org/10.1017/S0962492918000016)
