---
tags: [optimization, adjoint]
aliases: [Karush-Kuhn-Tucker conditions, First-order optimality]
---

For $\min_{\sigma,u}J(\sigma,u)$ subject to $e(\sigma,u) = 0$, with the [[Lagrangian Formulation|Lagrangian]] $\mathcal L = J+\langle\lambda,e\rangle$, a local minimiser $(\sigma^*,u^*)$ satisfies, under a constraint qualification (here: the state equation is uniquely solvable for every $\sigma$), the **first-order (KKT) conditions**:

1. $\partial_\lambda\mathcal L = 0$: the [[State Equation]] $e(\sigma,u) = 0$ (primal feasibility);
2. $\partial_u\mathcal L = 0$: the [[Adjoint Equation]];
3. $\partial_\sigma\mathcal L = 0$: the gradient vanishes.

With the box constraints $\sigma_{\min}\le\sigma\le\sigma_{\max}$, condition 3 becomes a variational inequality:

$$
\langle\partial_\sigma\mathcal L,\ \tilde\sigma-\sigma^*\rangle\ge0\qquad\forall\tilde\sigma\in\Sigma .
$$

Pointwise, the gradient is zero where the bound is inactive, non-negative at the lower bound and non-positive at the upper bound.

**Relation to the adjoint method.** An iterate is not a KKT point. The [[Adjoint State Method]] enforces conditions 1 and 2 exactly, by solving the state and adjoint equations, and *evaluates* the left-hand side of 3. The result equals the gradient of the reduced functional $\sigma\mapsto J(\sigma,u(\sigma))$, which the optimiser then drives to zero.

## References

1. J. Nocedal, S. J. Wright (2006). *Numerical Optimization*, 2nd ed. Springer. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
2. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
