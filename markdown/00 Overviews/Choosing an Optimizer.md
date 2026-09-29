---
tags: [overview, optimization]
---

Which method minimises $J(\sigma) = \Phi(\sigma)+\alpha R(\sigma)$ best depends mainly on two properties: whether the misfit $\Phi$ is a sum of squares, and whether the regulariser $R$ is smooth.

## Least-squares misfit, smooth regulariser

The [[Gauss-Newton Method]] uses the Jacobian of the residual. It converges in few iterations, typically a handful to a few dozen, because it captures the curvature of the misfit. Each iteration solves a linear system in the conductivity unknowns:

- with a dense matrix when there are few conductivity unknowns;
- through the Woodbury identity, in the space of the measurements, when there are more unknowns than measurements.

The [[Levenberg-Marquardt Method]] adds adaptive damping and makes the method robust far from the solution. A [[Line Search]] is the alternative. Smoothed total variation (see [[Smoothed Total Variation]]) fits into this framework through its lagged-diffusivity Hessian.

## Gradient only

When the Jacobian is too expensive (many measurements and many unknowns) or the misfit is not a sum of squares (e.g. the [[Kohn-Vogelius Functional]]), quasi-Newton methods need only gradients (see [[L-BFGS]]). The choice of inner product for the gradient matters: the $L^2$ gradient behaves the same on all meshes, the coefficient gradient does not (see [[Gradient Representation and the Riesz Map]]). Bounds are handled by projection (see [[L-BFGS-B]], [[Box Constraints on Conductivity]]). [[Stochastic and Adaptive Gradient Methods]] are mainly relevant for training learned components.

## Non-smooth or learned regulariser

Exact total variation, constraints, or a regulariser given only through a denoiser are handled through their [[Proximal Operator]]:

- **Proximal gradient.** It alternates gradient steps on Φ with proximal steps on $R$.
- **[[ADMM]].** It splits the problem so that the expensive, nonlinear data term and the cheap prior are treated separately (see [[Nested ADMM Reconstruction]]). The regulariser step can be exact TV via the [[Chambolle-Pock Algorithm]], or a learned denoiser (see [[Plug-and-Play Priors]]).

## Stopping

With noisy data, iterating to convergence overfits the noise. The discrepancy principle stops when the misfit reaches the expected noise level (see [[Stopping Criteria]], [[Choosing the Regularization Parameter]]).

## References

1. J. Nocedal, S. J. Wright (2006). *Numerical Optimization*, 2nd ed. Springer. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
2. M. Benning, M. Burger (2018). *Modern regularization methods for inverse problems*. Acta Numer. 27, 1–111. [doi:10.1017/S0962492918000016](https://doi.org/10.1017/S0962492918000016)
