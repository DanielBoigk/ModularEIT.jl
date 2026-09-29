---
tags: [optimization, adjoint]
---

EIT reconstruction is a **PDE-constrained optimisation** problem:

$$
\min_{\sigma\in\Sigma,\ u}\ J(u) + \beta\,\mathcal R(\sigma)
\qquad\text{subject to}\qquad e(\sigma,u) = 0,
$$

where $e(\sigma,u)=0$ is the (weak) [[Conductivity Equation]] with its boundary condition. $\sigma$ is the **control** or parameter and $u$ the **state**. The admissible set

$$
\Sigma = \{\sigma\in L^\infty(\Omega):\ \sigma_{\min}\le\sigma\le\sigma_{\max}\ \text{a.e.}\},\qquad 0<\sigma_{\min}\le\sigma_{\max}<\infty,
$$

guarantees a unique state for every $\sigma$ (see [[Lax-Milgram Theorem]]).

**Two viewpoints.**

- *All-at-once:* treat $(\sigma,u)$ as joint unknowns and enforce $e=0$ through Lagrange multipliers (see [[Lagrangian Formulation]] and [[KKT Conditions]]).
- *Reduced (black-box):* eliminate the state through the control-to-state map $u = u(\sigma)$ and minimise the reduced functional $\hat J(\sigma) = J(u(\sigma))+\beta\mathcal R(\sigma)$ over $\sigma$ alone.

The [[Adjoint State Method]] computes the gradient of the reduced functional at the cost of one extra linear solve per state, *independent of the number of parameters*. That is what makes pixel- or element-wise conductivity reconstructions feasible.

Discretisation can happen before or after deriving optimality conditions (see [[Discretize-then-Optimize vs Optimize-then-Discretize]]).

## References

1. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
2. F. Tröltzsch (2010). *Optimal Control of Partial Differential Equations*. AMS GSM 112. [doi:10.1090/gsm/112](https://doi.org/10.1090/gsm/112)
