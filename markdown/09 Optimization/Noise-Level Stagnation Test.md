---
tags: [optimization, stopping, noise]
aliases: [Significance stopping rule, Stagnation at the noise level]
---

A stopping rule for iterative reconstructions that measures each step's decrease of the misfit against the statistical fluctuation of the misfit itself. It stops once a step no longer makes significant progress, and returns the iterate before that step.

## The fluctuation of the misfit

Let $r(\sigma)\in\mathbb R^m$ be the whitened residual, so that the noise is $\mathcal N(0, I)$ (for correlated or model-dependent noise, after the whitening of the [[Approximation Error Approach]]), and $\Phi = \tfrac12\lVert r\rVert^2$. At the true conductivity, $2\Phi\sim\chi^2_m$, so

$$
\mathbb E\,\Phi = \frac m2,\qquad \operatorname{sd}\Phi = \sqrt{\frac m2}.
$$

The [[Choosing the Regularization Parameter|discrepancy principle]] uses the mean: stop at the first iterate with $\Phi_k\le\tau^2 m/2$. It relies on the noise model being right. When the covariance is estimated, as in the approximation error approach, the misfit of the true conductivity is only approximately $m/2$. The first crossing of $\tau^2 m/2$ can then come several iterations before the error is smallest, while the misfit is still decreasing substantially. Or it can never come, and the iteration runs into its budget.

## The test

Iterative methods for ill-posed problems are semi-convergent (see [[Implicit Regularization]]). Early steps resolve structure and decrease the misfit a lot. Late steps fit noise: the misfit barely changes, while the error grows, often quickly. A decrease below the fluctuation of the misfit cannot be told apart from fitting noise, so with a factor $\kappa$ (2 to 3):

$$
\text{accept step } k \text{ if } \Phi_{k-1}-\Phi_k \ge \kappa\sqrt{m/2};\quad\text{otherwise stop and return }\sigma_{k-1}.
$$

The rejected step costs one extra iteration, but never enters the result. Unlike the relative tolerances of the stagnation criteria in [[Stopping Criteria]], the threshold is absolute and set by the noise, not by the size of $\Phi$. The rule needs no target value, so it is unaffected by an imprecise covariance estimate. Using the $\chi^2$ distribution of the whitened misfit for parameter choice also underlies the $\chi^2$ principle of Mead and Renaut for Tikhonov regularisation.

**Robustness.** The decrease between successive iterates is not itself $\chi^2$-distributed, so $\kappa$ is a scale, not a significance level. The rule is reliable when the iteration shows a clear gap between the last significant and the first insignificant step. In a simulated EIT example with $m = 16\,384$ residuals and an approximation error model, Levenberg–Marquardt steps decreased $\Phi$ by $7.1$ and then $1.3$ standard deviations. Any $\kappa$ between these values returns the iterate with the smallest error, even though the discrepancy principle had stopped two iterations earlier. The next steps increased the error from $0.25$ to $0.72$ within five iterations.

**In ModularEIT.jl:** a `callback` of [`minimize`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.minimize), with the residual count from [`n_residual`](https://danielboigk.github.io/ModularEIT.jl/dev/api/objectives/#ModularEIT.n_residual).

## References

1. M. Hanke (1997). *A regularizing Levenberg–Marquardt scheme, with applications to inverse groundwater filtration problems*. Inverse Problems 13(1), 79–95. [doi:10.1088/0266-5611/13/1/007](https://doi.org/10.1088/0266-5611/13/1/007)
2. B. Kaltenbacher, A. Neubauer, O. Scherzer (2008). *Iterative Regularization Methods for Nonlinear Ill-Posed Problems*. De Gruyter. [doi:10.1515/9783110208276](https://doi.org/10.1515/9783110208276)
3. J. L. Mead, R. A. Renaut (2009). *A Newton root-finding algorithm for estimating the regularization parameter for solving ill-conditioned least squares problems*. Inverse Problems 25(2), 025002. [doi:10.1088/0266-5611/25/2/025002](https://doi.org/10.1088/0266-5611/25/2/025002)
4. J. Kaipio, E. Somersalo (2007). *Statistical inverse problems: Discretization, model reduction and inverse crimes*. J. Comput. Appl. Math. 198(2), 493–504. [doi:10.1016/j.cam.2005.09.027](https://doi.org/10.1016/j.cam.2005.09.027)
